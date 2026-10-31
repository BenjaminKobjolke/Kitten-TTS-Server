# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS ONNX.
# Supports multiple KittenTTS model variants with hot-swappable model switching.

import gc
import torch
import json as json
import logging
import threading
import sys
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List, Dict, Any
from huggingface_hub import hf_hub_download as hf_hub_download
import phonemizer

from config import config_manager as config_manager
from engine_registry import (
    MODEL_REGISTRY,
    _V01_VOICES,
    _V01_VOICE_ALIASES as _V01_VOICE_ALIASES,
    resolve_selector as resolve_selector,
    get_model_registry as _get_model_registry,
)
from engine_text import TextCleaner, basic_english_tokenize
from engine_espeak import initialize_espeak
from engine_info import get_model_info as _get_model_info
from engine_synthesis import synthesize as _synthesize
from engine_loader import load_model as _load_model

logger = logging.getLogger(__name__)
MODEL_TOKEN_LIMIT = 500

onnx_session: Optional[ort.InferenceSession] = None
voices_data: Optional[dict] = None
phonemizer_backend: Optional[phonemizer.backend.EspeakBackend] = None
text_cleaner: Optional["TextCleaner"] = None
MODEL_LOADED: bool = False

# Track which model is loaded
loaded_model_selector: Optional[str] = None
loaded_model_repo_id: Optional[str] = None
loaded_model_device: Optional[str] = None

# Voice alias mapping for v0.8 models (e.g. "Jasper" -> "expr-voice-2-m")
_voice_aliases: Dict[str, str] = {}
# Speed priors per voice from model config (per-voice speed correction factors)
_speed_priors: Dict[str, float] = {}

# --- Async Loading & Cancellation ---
_load_lock = threading.Lock()
_cancel_event = threading.Event()
_load_thread: Optional[threading.Thread] = None

# Download progress tracking for UI status modal
_download_status: Dict[str, Any] = {
    "active": False,
    "phase": "",
    "detail": "",
    "progress_pct": 0,
    "error": None,
}


def _check_cancelled():
    """Raises RuntimeError if model loading has been cancelled."""
    if _cancel_event.is_set():
        raise RuntimeError("Model loading cancelled by user.")


def _update_download_status(
    phase: str, detail: str = "", progress_pct: int = 0, error: str = None
):
    """Update the download status for the UI to poll."""
    global _download_status
    _download_status = {
        "active": error is None and phase != "complete",
        "phase": phase,
        "detail": detail,
        "progress_pct": progress_pct,
        "error": error,
    }


def get_download_status() -> Dict[str, Any]:
    """Returns the current download/loading status for UI polling."""
    return dict(_download_status)


def is_loading() -> bool:
    """Returns True if a model is currently being loaded in the background."""
    return _load_thread is not None and _load_thread.is_alive()


def get_available_voices() -> List[str]:
    """Returns the named voice list for the currently loaded model (for UI dropdown)."""
    if loaded_model_selector and loaded_model_selector in MODEL_REGISTRY:
        return list(MODEL_REGISTRY[loaded_model_selector]["voices"])
    # Fallback
    return list(_V01_VOICES)


def get_all_accepted_voices() -> List[str]:
    """Returns all accepted voice identifiers (named + internal) for API validation."""
    named = get_available_voices()
    # Also accept raw internal voice keys (expr-voice-*) for API compatibility
    if voices_data is not None:
        internal = list(voices_data.keys())
        return list(dict.fromkeys(named + internal))  # deduplicated, order preserved
    return named


def get_model_info() -> Dict[str, Any]:
    """Return information about the currently loaded model."""
    return _get_model_info(sys.modules[__name__])


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """Return the registry data used by the UI dropdown."""
    return _get_model_registry()


def _phoneme_token_chunks(phonemes: str) -> List[List[int]]:
    """Convert phonemes into model-sized token groups without splitting words."""
    if text_cleaner is None:
        raise RuntimeError("Text cleaner is not initialized")

    word_tokens = basic_english_tokenize(phonemes)
    chunks: List[List[int]] = []
    current_words: List[str] = []

    def append_chunk(words: List[str]) -> None:
        token_ids = text_cleaner(" ".join(words))
        chunks.append([0, *token_ids, 0])

    for word in word_tokens:
        candidate = current_words + [word]
        candidate_ids = text_cleaner(" ".join(candidate))
        if len(candidate_ids) + 2 <= MODEL_TOKEN_LIMIT:
            current_words = candidate
            continue

        if current_words:
            append_chunk(current_words)
            current_words = []

        word_ids = text_cleaner(word)
        if len(word_ids) + 2 <= MODEL_TOKEN_LIMIT:
            current_words = [word]
            continue

        # A single unbroken token has no word boundary to use; split its symbols
        # so the model still never receives an over-limit sequence.
        symbol_limit = MODEL_TOKEN_LIMIT - 2
        chunks.extend(
            [[0, *word_ids[i : i + symbol_limit], 0]]
            for i in range(0, len(word_ids), symbol_limit)
        )

    if current_words:
        append_chunk(current_words)
    return chunks


def _init_espeak():
    initialize_espeak(sys.modules[__name__])


def load_model() -> bool:
    """Load the configured KittenTTS model."""
    return _load_model(sys.modules[__name__])


def unload_model() -> bool:
    """
    Unloads the current model and releases resources.
    Does NOT reload - use reload_model() for hot-swap.

    Returns:
        bool: True if the model was unloaded successfully.
    """
    global onnx_session, voices_data, MODEL_LOADED
    global loaded_model_selector, loaded_model_repo_id, loaded_model_device
    global _voice_aliases, _speed_priors

    logger.info("Initiating model unload sequence...")

    if onnx_session is not None:
        logger.info("Unloading ONNX session from memory...")
        del onnx_session
        onnx_session = None

    if voices_data is not None:
        del voices_data
        voices_data = None

    MODEL_LOADED = False
    loaded_model_selector = None
    loaded_model_repo_id = None
    loaded_model_device = None
    _voice_aliases = {}
    _speed_priors = {}

    # Force garbage collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # Clear GPU cache if available
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    logger.info("Model unloaded and resources released.")
    return True


def _reload_model_worker():
    """
    Background worker that performs the actual model reload.
    Runs in a separate thread so the FastAPI server stays responsive.
    """
    try:
        _update_download_status("unloading", "Unloading current model...", 5)
        unload_model()

        _check_cancelled()
        logger.info("Resources cleared. Loading model from updated config...")
        load_model()
    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Reload worker cancelled: {e}")
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
        else:
            logger.error(f"Reload worker error: {e}", exc_info=True)
            _update_download_status("error", "", 0, str(e))
    except Exception as e:
        logger.error(f"Reload worker error: {e}", exc_info=True)
        _update_download_status("error", "", 0, str(e))


def reload_model_async():
    """
    Initiates a model hot-swap in a background thread.
    Returns immediately so the server can continue serving status polls.
    If a load is already in progress, cancels it first.
    """
    global _load_thread

    logger.info("Initiating async model hot-swap/reload sequence...")

    # Cancel any in-progress load
    if is_loading():
        logger.info("Cancelling in-progress model load...")
        _cancel_event.set()
        _load_thread.join(timeout=15)
        logger.info("Previous load thread finished.")

    # Reset cancel flag for new load
    _cancel_event.clear()

    # Start new background load
    _load_thread = threading.Thread(target=_reload_model_worker, daemon=True)
    _load_thread.start()
    logger.info("Background model reload thread started.")


def cancel_loading():
    """Cancels any in-progress model loading."""
    if is_loading():
        logger.info("Cancelling model loading by user request...")
        _cancel_event.set()
        _update_download_status("cancelling", "Cancelling model load...", 0)
        return True
    return False


def reload_model() -> bool:
    """
    Synchronous reload (used for startup only).
    Unloads current model, clears resources, reloads from config.
    """
    logger.info("Initiating synchronous model reload...")
    _cancel_event.clear()
    _update_download_status("unloading", "Unloading current model...", 5)
    unload_model()
    logger.info("Resources cleared. Loading model from updated config...")
    return load_model()


def synthesize(
    text: str, voice: str, speed: float = 1.0, *, raise_on_error: bool = False
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Synthesize text while preserving the engine module API."""
    return _synthesize(
        sys.modules[__name__], text, voice, speed, raise_on_error=raise_on_error
    )


# --- End File: engine.py ---
