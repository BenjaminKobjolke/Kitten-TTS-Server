"""Model metadata and selector lookup for KittenTTS models."""

import logging
from typing import Any, Dict

logger = logging.getLogger("engine")

# v0.1/v0.2 models use named voices mapped to internal expr-voice-* keys.
_V01_VOICES = ["Amber", "Felix", "Clara", "Marcus", "Ivy", "Oscar", "Nora", "Reed"]
_V01_VOICE_ALIASES = {
    "Amber": "expr-voice-2-f",
    "Felix": "expr-voice-2-m",
    "Clara": "expr-voice-3-f",
    "Marcus": "expr-voice-3-m",
    "Ivy": "expr-voice-4-f",
    "Oscar": "expr-voice-4-m",
    "Nora": "expr-voice-5-f",
    "Reed": "expr-voice-5-m",
}
_V08_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "kitten-nano-0.1": {
        "repo_id": "KittenML/kitten-tts-nano-0.1",
        "display_name": "Nano 0.1 (15M, Original)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "15M",
        "size": "<25MB",
        "version": "0.1",
    },
    "kitten-nano-0.2": {
        "repo_id": "KittenML/kitten-tts-nano-0.2",
        "display_name": "Nano 0.2 (15M, Preview)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "15M",
        "size": "<25MB",
        "version": "0.2",
    },
    "kitten-mini-0.1": {
        "repo_id": "KittenML/kitten-tts-mini-0.1",
        "display_name": "Mini 0.1 (80M, Original)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "80M",
        "size": "~170MB",
        "version": "0.1",
    },
    "kitten-nano-0.8-int8": {
        "repo_id": "KittenML/kitten-tts-nano-0.8-int8",
        "display_name": "Nano 0.8 INT8 (15M, Quantized)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "15M",
        "size": "~25MB",
        "version": "0.8",
    },
    "kitten-nano-0.8-fp32": {
        "repo_id": "KittenML/kitten-tts-nano-0.8-fp32",
        "display_name": "Nano 0.8 FP32 (15M, Full Precision)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "15M",
        "size": "~50MB",
        "version": "0.8",
    },
    "kitten-micro-0.8": {
        "repo_id": "KittenML/kitten-tts-micro-0.8",
        "display_name": "Micro 0.8 (40M, Balanced)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "40M",
        "size": "~40MB",
        "version": "0.8",
    },
    "kitten-mini-0.8": {
        "repo_id": "KittenML/kitten-tts-mini-0.8",
        "display_name": "Mini 0.8 (80M, Highest Quality)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "80M",
        "size": "~79MB",
        "version": "0.8",
    },
}

_REPO_TO_SELECTOR = {
    info["repo_id"]: selector for selector, info in MODEL_REGISTRY.items()
}


def resolve_selector(config_value: str) -> str:
    """Resolve a selector or Hugging Face repo ID to a registry key."""
    if config_value in MODEL_REGISTRY:
        return config_value
    if config_value in _REPO_TO_SELECTOR:
        return _REPO_TO_SELECTOR[config_value]

    lower = config_value.lower().strip()
    for selector in MODEL_REGISTRY:
        if selector.lower() == lower:
            return selector
    for repo_id, selector in _REPO_TO_SELECTOR.items():
        if repo_id.lower() == lower:
            return selector

    logger.warning(
        f"Unknown model selector '{config_value}'. "
        f"Valid selectors: {list(MODEL_REGISTRY.keys())}. "
        "Defaulting to 'kitten-nano-0.1'."
    )
    return "kitten-nano-0.1"


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """Return model metadata for the UI dropdown."""
    return {
        key: {
            "display_name": value["display_name"],
            "voices": value["voices"],
            "default_voice": value["default_voice"],
            "params": value["params"],
            "size": value["size"],
            "version": value["version"],
        }
        for key, value in MODEL_REGISTRY.items()
    }
