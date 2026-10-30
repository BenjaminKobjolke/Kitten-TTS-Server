"""Typed accessors for values held by the configuration manager."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("config")


class ConfigValueGetters:
    """Convert stored configuration values to common Python types."""

    def get_int(self, key_path: str, default: Optional[int] = None) -> int:
        """Retrieves a configuration value, converting it to an integer."""
        raw_value = self.get(key_path)
        if raw_value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' is None, using provided method default: {default}"
                )
                return default
            logger.error(
                f"Mandatory integer config '{key_path}' is None, and no method default. Returning 0."
            )
            return 0
        try:
            return int(raw_value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{raw_value}' for '{key_path}'. Using method default or 0."
            )
            if isinstance(default, int):
                return default
            logger.error(
                f"Cannot parse '{raw_value}' as int for '{key_path}' and no valid method default. Returning 0."
            )
            return 0

    def get_float(self, key_path: str, default: Optional[float] = None) -> float:
        """Retrieves a configuration value, converting it to a float."""
        raw_value = self.get(key_path)
        if raw_value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' is None, using provided method default: {default}"
                )
                return default
            logger.error(
                f"Mandatory float config '{key_path}' is None, and no method default. Returning 0.0."
            )
            return 0.0
        try:
            return float(raw_value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{raw_value}' for '{key_path}'. Using method default or 0.0."
            )
            if isinstance(default, float):
                return default
            logger.error(
                f"Cannot parse '{raw_value}' as float for '{key_path}' and no valid method default. Returning 0.0."
            )
            return 0.0

    def get_bool(self, key_path: str, default: Optional[bool] = None) -> bool:
        """Retrieves a configuration value, converting it to a boolean."""
        raw_value = self.get(key_path)
        if raw_value is None:
            if default is not None:
                logger.debug(
                    f"Config '{key_path}' is None, using provided method default: {default}"
                )
                return default
            logger.error(
                f"Mandatory boolean config '{key_path}' is None, and no method default. Returning False."
            )
            return False
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            normalized_value = raw_value.lower()
            if normalized_value in ("true", "1", "t", "yes", "y"):
                return True
            if normalized_value in ("false", "0", "f", "no", "n"):
                return False
            logger.warning(
                f"Invalid boolean value '{raw_value}' for '{key_path}'. Using method default or False."
            )
            if default is not None:
                return default
            logger.error(
                f"Cannot parse '{raw_value}' as bool for '{key_path}' and no method default. Returning False."
            )
            return False
        try:  # Handle numeric representations (e.g., 1 for True, 0 for False).
            return bool(int(raw_value))
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid boolean value '{raw_value}' for '{key_path}'. Using method default or False."
            )
            if isinstance(default, bool):
                return default
            logger.error(
                f"Cannot parse '{raw_value}' as bool for '{key_path}' and no valid method default. Returning False."
            )
            return False

    def get_path(
        self,
        key_path: str,
        default_str_path: Optional[str] = None,
        ensure_absolute: bool = False,
    ) -> Path:
        """
        Retrieves a configuration value expected to be a path, returning it as a Path object.
        If 'ensure_absolute' is True, the path is resolved to an absolute path.
        """
        value_from_config = self.get(key_path)

        path_obj_to_return: Path
        if isinstance(value_from_config, Path):
            path_obj_to_return = value_from_config
        elif isinstance(value_from_config, str):  # Convert string from config to Path.
            path_obj_to_return = Path(value_from_config)
        elif default_str_path is not None:  # Fallback to provided string default.
            logger.debug(
                f"Config Path '{key_path}' not found or invalid type, using provided default string path: '{default_str_path}'"
            )
            path_obj_to_return = Path(default_str_path)
        else:  # Ultimate fallback if no value and no default.
            logger.error(
                f"Config Path '{key_path}' not found or invalid type, and no default provided. Returning Path('.')"
            )
            path_obj_to_return = Path(".")  # Current directory.

        return path_obj_to_return.resolve() if ensure_absolute else path_obj_to_return
