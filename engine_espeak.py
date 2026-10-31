"""eSpeak backend setup used by the KittenTTS engine."""

import os
from pathlib import Path

import phonemizer


def initialize_espeak(engine):
    """Initialize eSpeak and the engine's text cleaner once."""
    if engine.phonemizer_backend is not None:
        return

    if os.name == "nt":
        engine.logger.info("Checking for eSpeak NG on Windows...")
        possible_paths = [
            Path(r"C:\Program Files\eSpeak NG"),
            Path(r"C:\Program Files (x86)\eSpeak NG"),
            Path(r"C:\eSpeak NG"),
            Path(os.environ.get("ProgramFiles", "")) / "eSpeak NG",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "eSpeak NG",
        ]
        espeak_found = False
        for espeak_path in possible_paths:
            if espeak_path.exists():
                dll_path = espeak_path / "libespeak-ng.dll"
                if dll_path.exists():
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll_path)
                    from phonemizer.backend.espeak.wrapper import (
                        EspeakWrapper as PhonemizeEspeakWrapper,
                    )

                    PhonemizeEspeakWrapper.set_library(str(dll_path))
                    engine.logger.info(f"Auto-configured eSpeak from: {espeak_path}")
                    espeak_found = True
                    break
        if not espeak_found:
            engine.logger.warning("eSpeak NG not found in common Windows locations.")

    elif os.name == "posix":
        engine.logger.info("Checking for system-installed eSpeak NG on Linux...")
        espeak_lib_path = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
        if Path(espeak_lib_path).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path
            engine.logger.info(
                f"Found and configured system eSpeak NG library: {espeak_lib_path}"
            )
        else:
            engine.logger.warning(
                f"Could not find system eSpeak NG library at {espeak_lib_path}. "
                "Please ensure 'espeak-ng' is installed via your package manager."
            )

    try:
        import logging as log_module

        phonemizer_logger = log_module.getLogger("phonemizer")
        original_level = phonemizer_logger.level
        phonemizer_logger.setLevel(log_module.ERROR)
        engine.phonemizer_backend = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        phonemizer_logger.setLevel(original_level)
        engine.logger.info("Phonemizer backend initialized successfully")
    except Exception as error:
        engine.logger.error(f"Failed to initialize phonemizer: {error}")
        engine.logger.error(
            "Please ensure eSpeak NG is installed:\n"
            "  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases\n"
            "  Linux: Run 'sudo apt install espeak-ng'"
        )
        raise

    engine.text_cleaner = engine.TextCleaner()
