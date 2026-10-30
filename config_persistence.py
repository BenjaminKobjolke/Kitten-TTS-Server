"""Thread-safe YAML configuration loading and saving operations."""

import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from config_defaults import (
    CONFIG_FILE_PATH,
    _deep_merge_dicts,
    _get_nested_value,
    _set_nested_value,
)

logger = logging.getLogger("config")


class ConfigPersistence:
    """Load and persist YAML configuration for a manager instance."""

    def _prepare_config_for_saving(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares a configuration dictionary for YAML serialization by converting
        internal Path objects back to their string representations.
        Returns a deep copy of the configuration dictionary with paths as strings.
        """
        config_copy_for_saving = deepcopy(config_dict)
        path_key_map_for_conversion = {
            "server": ["log_file_path"],
            "paths": ["model_cache", "output"],
        }
        for section, keys_list in path_key_map_for_conversion.items():
            if section in config_copy_for_saving:
                for key in keys_list:
                    current_path_val = _get_nested_value(
                        config_copy_for_saving, [section, key]
                    )
                    if isinstance(current_path_val, Path):
                        _set_nested_value(
                            config_copy_for_saving,
                            [section, key],
                            str(current_path_val),
                        )
        return config_copy_for_saving

    def load_config(self):
        """
        Loads the application configuration from 'config.yaml'.
        If the file doesn't exist, it's created using defaults.
        The loaded configuration is merged with defaults to ensure all expected
        keys are present.
        Device settings and path types are resolved after loading.
        """
        with self._lock:  # Ensure thread-safe loading.
            base_defaults = self._load_defaults()  # Ensures default paths exist.

            if CONFIG_FILE_PATH.exists():
                logger.info(f"Loading configuration from: {CONFIG_FILE_PATH}")
                try:
                    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f)
                    if isinstance(yaml_data, dict):
                        # Merge loaded YAML data into a copy of defaults.
                        # YAML data takes precedence, defaults fill in missing parts.
                        effective_config = deepcopy(base_defaults)
                        _deep_merge_dicts(yaml_data, effective_config)
                        self.config = effective_config
                        logger.info(
                            "Successfully loaded and merged configuration from "
                            f"{CONFIG_FILE_PATH}."
                        )
                    else:
                        logger.error(
                            f"Invalid format in {CONFIG_FILE_PATH}. Expected a "
                            "dictionary. Using defaults and attempting to overwrite "
                            "the invalid file."
                        )
                        self.config = base_defaults  # Fallback to defaults.
                        if not self._save_config_yaml_internal(self.config):
                            logger.error(
                                "Failed to overwrite invalid "
                                f"{CONFIG_FILE_PATH} with defaults."
                            )
                except yaml.YAMLError as e:
                    logger.error(
                        f"Error parsing YAML from {CONFIG_FILE_PATH}: {e}. "
                        "Using defaults and attempting to overwrite the corrupted "
                        "file."
                    )
                    self.config = base_defaults
                    if not self._save_config_yaml_internal(self.config):
                        logger.error(
                            "Failed to overwrite corrupted "
                            f"{CONFIG_FILE_PATH} with defaults."
                        )
                except Exception as e:
                    logger.error(
                        f"Unexpected error loading {CONFIG_FILE_PATH}: {e}. "
                        "Using in-memory defaults.",
                        exc_info=True,
                    )
                    self.config = base_defaults  # Use defaults; don't save after
                    # unexpected errors.
            else:
                logger.info(
                    f"{CONFIG_FILE_PATH} not found. Creating initial configuration "
                    "using defaults..."
                )
                # Start with defaults.
                self.config = base_defaults
                if self._save_config_yaml_internal(self.config):
                    logger.info(
                        "Successfully created and saved initial default configuration "
                        f"to {CONFIG_FILE_PATH}."
                    )
                else:
                    logger.error(
                        f"Failed to save initial configuration to {CONFIG_FILE_PATH}. "
                        f"Using in-memory defaults."
                    )

            # Resolve the device and convert path strings to Path objects for the
            # loaded or created config.
            self.config = self._resolve_paths_and_device(self.config)
            logger.debug(f"Current configuration loaded and resolved: {self.config}")
            return self.config

    def _save_config_yaml_internal(self, config_dict_to_save: Dict[str, Any]) -> bool:
        """
        Internal method to save the provided configuration dictionary to 'config.yaml'.
        It includes a backup and restore mechanism for safety during writes.
        Assumes the caller holds the necessary lock.
        Converts Path objects to strings before YAML serialization.
        """
        # Prepare the configuration for saving (e.g., convert Path objects to strings).
        prepared_config_for_yaml = self._prepare_config_for_saving(config_dict_to_save)

        temp_file = CONFIG_FILE_PATH.with_suffix(CONFIG_FILE_PATH.suffix + ".tmp")
        backup_file = CONFIG_FILE_PATH.with_suffix(CONFIG_FILE_PATH.suffix + ".bak")

        try:
            # Atomically write to a temporary file first.
            with open(temp_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    prepared_config_for_yaml,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )

            # If an existing config file exists, back it up.
            if CONFIG_FILE_PATH.exists():
                try:
                    shutil.move(str(CONFIG_FILE_PATH), str(backup_file))
                    logger.debug(f"Backed up existing configuration to {backup_file}")
                except Exception as backup_error:
                    logger.warning(
                        f"Could not create backup of {CONFIG_FILE_PATH}: {backup_error}"
                    )
                    # Proceed with saving, but warn about missing backup.

            # Rename the temporary file to the actual configuration file.
            shutil.move(str(temp_file), str(CONFIG_FILE_PATH))
            logger.info(f"Configuration successfully saved to {CONFIG_FILE_PATH}")
            return True

        except yaml.YAMLError as e_yaml:
            logger.error(
                f"Error formatting data for {CONFIG_FILE_PATH} (YAML error): {e_yaml}",
                exc_info=True,
            )
            return False
        except Exception as e_general:
            logger.error(
                f"Failed to save configuration to {CONFIG_FILE_PATH}: {e_general}",
                exc_info=True,
            )
            # Attempt to restore from backup if the save operation failed.
            if backup_file.exists() and not CONFIG_FILE_PATH.exists():
                try:
                    shutil.move(str(backup_file), str(CONFIG_FILE_PATH))
                    logger.info(
                        f"Restored configuration from backup {backup_file} due to "
                        "save failure."
                    )
                except Exception as restore_error:
                    logger.error(
                        f"Failed to restore configuration from backup: {restore_error}"
                    )
            # Clean up the temporary file if it still exists after a failure.
            if temp_file.exists():
                try:
                    os.remove(str(temp_file))
                except Exception as remove_error:
                    logger.warning(
                        "Could not remove temporary config file "
                        f"{temp_file}: {remove_error}"
                    )
            return False
        finally:
            # Clean up the backup file if the main configuration file exists and the save was successful.
            if CONFIG_FILE_PATH.exists() and backup_file.exists():
                try:
                    if (
                        CONFIG_FILE_PATH.stat().st_size > 0
                    ):  # Basic check that the new file is not empty.
                        os.remove(str(backup_file))
                        logger.debug(
                            f"Removed backup file {backup_file} after successful save."
                        )
                except Exception as remove_bak_error:
                    logger.warning(
                        f"Could not remove backup config file {backup_file}: {remove_bak_error}"
                    )

    def save_config_yaml(self) -> bool:
        """
        Public method to save the current in-memory configuration to 'config.yaml'.
        Ensures thread-safety using a lock.
        """
        with self._lock:
            return self._save_config_yaml_internal(self.config)
