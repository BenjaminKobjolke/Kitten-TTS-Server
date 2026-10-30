"""YAML-backed configuration lifecycle and persistence."""

import logging
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, Optional

import torch
from pathlib import Path

from config_defaults import (
    DEFAULT_CONFIG,
    _deep_merge_dicts,
    _ensure_default_paths_exist,
    _get_nested_value,
    _set_nested_value,
)
from config_persistence import ConfigPersistence
from config_value_getters import ConfigValueGetters

logger = logging.getLogger("config")


class YamlConfigManager(ConfigPersistence, ConfigValueGetters):
    """
    Manages application configuration stored in a YAML file.
    This class handles loading, saving, updating, and resetting the configuration.
    Operations are thread-safe for file writing.
    """

    def __init__(self):
        """Initializes the configuration manager by loading configuration from YAML."""
        self.config: Dict[str, Any] = {}
        self._lock = Lock()  # Ensures thread-safety for file write operations.
        self.load_config()

    def _load_defaults(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the hardcoded default configuration structure.
        Also ensures that default directory paths defined in the structure exist.
        """
        _ensure_default_paths_exist()  # Create necessary default directories.
        return deepcopy(DEFAULT_CONFIG)

    def _resolve_paths_and_device(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves device settings (e.g., 'auto' to 'cuda' or 'cpu') and converts
        string paths in the configuration data to Path objects for internal use.
        The 'config_data' dictionary is modified in place.
        """
        # Resolve TTS device setting with robust CUDA detection.
        current_device_setting = _get_nested_value(
            config_data, ["tts_engine", "device"], "auto"
        )
        if current_device_setting == "auto":
            resolved_device = self._detect_best_device()
            _set_nested_value(config_data, ["tts_engine", "device"], resolved_device)
        elif current_device_setting not in ["cuda", "cpu", "gpu"]:
            logger.warning(
                f"Invalid TTS device '{current_device_setting}' in configuration. "
                "Recognized values are 'auto', 'cpu', 'cuda', 'gpu'. "
                "Defaulting to auto-detection."
            )
            resolved_device = self._detect_best_device()
            _set_nested_value(config_data, ["tts_engine", "device"], resolved_device)

        final_device = _get_nested_value(config_data, ["tts_engine", "device"])
        logger.info(f"TTS processing device resolved to: {final_device}")

        # Convert relevant string paths to Path objects.
        path_key_map_for_conversion = {
            "server": ["log_file_path"],
            "paths": ["model_cache", "output"],
        }
        for section, keys_list in path_key_map_for_conversion.items():
            if section in config_data:
                for key in keys_list:
                    current_path_val = _get_nested_value(config_data, [section, key])
                    if isinstance(current_path_val, str):
                        _set_nested_value(
                            config_data, [section, key], Path(current_path_val)
                        )
        return config_data

    def _detect_best_device(self) -> str:
        """
        Robustly detects the best available device for TTS processing.
        Tests actual CUDA functionality rather than just checking availability.

        Returns:
            str: 'cuda' if CUDA is truly functional, 'cpu' otherwise.
        """
        # Test CUDA first as it's generally preferred for ML workloads
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality by creating a tensor and moving it to CUDA.
                test_tensor = torch.tensor([1.0])
                test_tensor = test_tensor.cuda()
                test_tensor = test_tensor.cpu()  # Clean up
                logger.info("CUDA test successful. Using CUDA device.")
                return "cuda"
            except Exception as e:
                logger.warning(
                    "CUDA is reported as available but failed functionality test: "
                    f"{e}. "
                    f"This usually means PyTorch was not compiled with CUDA support."
                )

        logger.info("CUDA not available or functional. Using CPU.")
        return "cpu"

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key path
        (e.g., 'server.port').
        If the key path is not found, 'default' is returned.
        For mutable types (dicts, lists), a deep copy is returned to prevent
        unintended modification of the in-memory configuration.
        """
        keys = key_path.split(".")
        with self._lock:  # Ensure thread-safe access to self.config.
            value = _get_nested_value(self.config, keys, default)
        return deepcopy(value) if isinstance(value, (dict, list)) else value

    def get_string(self, key_path: str, default: Optional[str] = None) -> str:
        """Retrieves a configuration value, ensuring it's a string."""
        # Added this method for explicit string retrieval, common for paths/IDs.
        raw_value = self.get(key_path)
        if raw_value is None:
            if default is not None:
                logger.debug(
                    f"Config string '{key_path}' is None, using provided method "
                    f"default: '{default}'"
                )
                return default
            logger.error(
                f"Mandatory string config '{key_path}' is None, and no method "
                "default. Returning empty string."
            )
            return ""
        if isinstance(
            raw_value, (Path, str)
        ):  # Handle Path objects by converting to string
            return str(raw_value)
        try:  # Attempt conversion for other types if necessary
            return str(raw_value)
        except Exception:
            logger.warning(
                f"Could not convert value '{raw_value}' for '{key_path}' to "
                "string. Using method default or empty string."
            )
            if default is not None:
                return default
            return ""

    def get_all(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the entire current configuration.
        Ensures thread-safety during the copy operation.
        """
        with self._lock:
            return deepcopy(self.config)

    def update_and_save(self, partial_update_dict: Dict[str, Any]) -> bool:
        """
        Deeply merges a 'partial_update_dict' into the current configuration
        and saves the entire updated configuration back to the YAML file.
        This allows updating specific nested values without overwriting entire sections.
        """
        if not isinstance(partial_update_dict, dict):
            logger.error("Invalid partial update data: input must be a dictionary.")
            return False

        with self._lock:
            try:
                # Work on a deep copy of the current config to avoid altering it
                # before a successful save.
                config_copy_for_update = deepcopy(self.config)
                # Merge the partial update into this copy.
                _deep_merge_dicts(partial_update_dict, config_copy_for_update)

                # Before saving, the merged config might need path/device re-resolution
                # if those specific keys were part of partial_update_dict.
                # For robustness, always re-resolve.
                resolved_updated_config = self._resolve_paths_and_device(
                    config_copy_for_update
                )

                if self._save_config_yaml_internal(resolved_updated_config):
                    # If save was successful, update the active in-memory config.
                    self.config = resolved_updated_config
                    logger.info(
                        "Configuration updated, saved, and re-resolved successfully."
                    )
                    return True
                else:
                    logger.error("Failed to save updated configuration after merging.")
                    return False
            except Exception as e:
                logger.error(
                    f"Error during configuration update and save process: {e}",
                    exc_info=True,
                )
                return False

    def reset_and_save(self) -> bool:
        """
        Resets the application configuration to its hardcoded defaults.
        The reset configuration (after resolving paths/device) is then saved
        to 'config.yaml'.
        """
        with self._lock:
            logger.warning("Initiating configuration reset to hardcoded defaults...")
            # Start with hardcoded defaults (this also ensures default
            # directories are created).
            reset_config_base = self._load_defaults()
            # Resolve device settings and ensure paths are Path objects for the
            # new in-memory config.
            final_reset_config = self._resolve_paths_and_device(reset_config_base)

            if self._save_config_yaml_internal(
                final_reset_config
            ):  # Save the fully resolved reset config.
                self.config = final_reset_config  # Update the active in-memory config.
                logger.info(
                    "Configuration successfully reset to defaults, saved, and resolved."
                )
                return True
            else:
                logger.error(
                    "Failed to save the reset configuration. Current configuration "
                    "remains unchanged."
                )
                # If save failed, the old self.config is retained.
                return False


config_manager = YamlConfigManager()
