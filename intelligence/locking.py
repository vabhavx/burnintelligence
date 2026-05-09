"""
PID lock for the intelligence pipeline. Prevents two continuous instances
from racing on the SQLite DB.
"""

import fcntl
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("intelligence.locking")

LOCK_DIR = Path.home() / ".burn_state"
LOCK_PATH = LOCK_DIR / "intel.lock"

_lock_fd = None


def acquire_lock_or_exit() -> int:
    """Acquire an exclusive non-blocking flock. Exits the process if held."""
    global _lock_fd

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_RDWR | os.O_CREAT, 0o644)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        logger.error("another instance running")
        sys.exit(1)

    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)

    _lock_fd = fd
    return fd


def release_lock() -> None:
    global _lock_fd
    if _lock_fd is None:
        return
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        os.close(_lock_fd)
    except OSError:
        pass
    finally:
        _lock_fd = None
        try:
            os.unlink(str(LOCK_PATH))
        except OSError:
            pass
