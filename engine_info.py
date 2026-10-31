"""Model state response data for KittenTTS API consumers."""

from typing import Any, Dict


def get_model_info(engine) -> Dict[str, Any]:
    """Return information about the engine's currently loaded model."""
    if (
        engine.loaded_model_selector
        and engine.loaded_model_selector in engine.MODEL_REGISTRY
    ):
        reg = engine.MODEL_REGISTRY[engine.loaded_model_selector]
        return {
            "loaded": engine.MODEL_LOADED,
            "selector": engine.loaded_model_selector,
            "repo_id": engine.loaded_model_repo_id,
            "display_name": reg["display_name"],
            "voices": reg["voices"],
            "default_voice": reg["default_voice"],
            "params": reg["params"],
            "size": reg["size"],
            "version": reg["version"],
            "device": engine.loaded_model_device,
        }
    return {
        "loaded": engine.MODEL_LOADED,
        "selector": engine.loaded_model_selector,
        "repo_id": engine.loaded_model_repo_id,
        "display_name": None,
        "voices": list(engine._V01_VOICES),
        "default_voice": "expr-voice-5-m",
        "params": None,
        "size": None,
        "version": None,
        "device": engine.loaded_model_device,
    }
