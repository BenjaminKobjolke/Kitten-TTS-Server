"""Module-level accessors for common application configuration values."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from config_defaults import DEFAULT_CONFIG, _get_nested_value
from config_manager import config_manager


# --- Convenience Accessor Functions ---
# These functions provide easy, module-level access to common configuration settings
# using the singleton 'config_manager' instance.


def _get_default_from_structure(key_path: str) -> Any:
    """Retrieve a default value directly from the default config structure."""
    keys = key_path.split(".")
    return _get_nested_value(DEFAULT_CONFIG, keys)


# Server Settings Accessors
def get_host() -> str:
    """Returns the server host address."""
    return config_manager.get_string(
        "server.host", _get_default_from_structure("server.host")
    )


def get_port() -> int:
    """Returns the server port number."""
    return config_manager.get_int(
        "server.port", _get_default_from_structure("server.port")
    )


# Audio Output Settings Accessors
def get_audio_output_format() -> str:
    """Returns the default audio output format (e.g., 'wav')."""
    return config_manager.get_string(
        "audio_output.format", _get_default_from_structure("audio_output.format")
    )


def get_log_file_path() -> Path:
    """Returns the absolute path to the server log file."""
    default_path_str = str(_get_default_from_structure("server.log_file_path"))
    return config_manager.get_path(
        "server.log_file_path", default_path_str, ensure_absolute=True
    )


# Model Settings Accessors
def get_model_repo_id() -> str:
    """Returns the Hugging Face repository ID for the model."""
    return config_manager.get_string(
        "model.repo_id", _get_default_from_structure("model.repo_id")
    )


# TTS Engine Settings Accessors
def get_tts_device() -> str:
    """Returns the resolved TTS processing device ('cuda' or 'cpu')."""
    # Device is resolved during load_config, so direct get is appropriate.
    return config_manager.get_string(
        "tts_engine.device", _get_default_from_structure("tts_engine.device")
    )


# General Path Settings Accessors
def get_model_cache_path(ensure_absolute: bool = True) -> Path:
    """Returns the path to the model cache directory."""
    default_path_str = str(_get_default_from_structure("paths.model_cache"))
    return config_manager.get_path(
        "paths.model_cache", default_path_str, ensure_absolute=ensure_absolute
    )


def get_output_path(ensure_absolute: bool = True) -> Path:
    """Returns the path to the default output directory."""
    default_path_str = str(_get_default_from_structure("paths.output"))
    return config_manager.get_path(
        "paths.output", default_path_str, ensure_absolute=ensure_absolute
    )


# Default Generation Parameter Accessors
def get_gen_default_speed() -> float:
    """Returns the default speed for TTS generation."""
    return config_manager.get_float(
        "generation_defaults.speed",
        _get_default_from_structure("generation_defaults.speed"),
    )


def get_gen_default_language() -> str:
    """Returns the default language for TTS generation."""
    return config_manager.get_string(
        "generation_defaults.language",
        _get_default_from_structure("generation_defaults.language"),
    )


# Audio Output Settings Accessors
def get_audio_sample_rate() -> int:
    """Returns the default audio sample rate."""
    return config_manager.get_int(
        "audio_output.sample_rate",
        _get_default_from_structure("audio_output.sample_rate"),
    )


# UI State Accessors
def get_ui_state() -> Dict[str, Any]:
    """Returns the entire UI state dictionary (for UI persistence)."""
    return config_manager.get(
        "ui_state", deepcopy(_get_default_from_structure("ui_state"))
    )


# General UI Settings Accessors
def get_ui_title() -> str:
    """Returns the title for the web UI."""
    return config_manager.get_string(
        "ui.title", _get_default_from_structure("ui.title")
    )


def get_full_config_for_template() -> Dict[str, Any]:
    """
    Returns a deep copy of the current configuration, with Path objects
    converted to strings. This is suitable for serialization (e.g., JSON)
    or for passing to web templates or API responses.
    """
    config_snapshot = config_manager.get_all()  # Gets a deep copy.
    # Convert Path objects in this snapshot to strings for serialization.
    return config_manager._prepare_config_for_saving(config_snapshot)


# --- End File: config.py ---
