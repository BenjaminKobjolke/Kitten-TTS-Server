# shutdown_watcher.py
# Stops the server gracefully when tools/shutdown.bat drops a _shutdown flag file.

import logging
import signal
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SHUTDOWN_FLAG_FILE = Path(__file__).parent / "_shutdown"


def check_shutdown_flag(flag: Path) -> bool:
    """Return True and consume the flag if it exists."""
    if not flag.exists():
        return False
    flag.unlink(missing_ok=True)
    return True


def _watch(flag: Path, interval: float) -> None:
    while not check_shutdown_flag(flag):
        time.sleep(interval)
    logger.info(f"Shutdown flag {flag} found. Stopping server...")
    # Runs uvicorn's own SIGINT handler like Ctrl+C; os.kill would hard-terminate on Windows.
    signal.raise_signal(signal.SIGINT)


def start_shutdown_watcher(flag: Path = SHUTDOWN_FLAG_FILE, interval: float = 1.0) -> None:
    """Start a daemon thread that polls for the shutdown flag file."""
    # A flag left over from an earlier run would stop the server right after startup.
    check_shutdown_flag(flag)
    threading.Thread(target=_watch, args=(flag, interval), daemon=True).start()
    logger.info(f"Shutdown watcher started. Create {flag} to stop the server.")
