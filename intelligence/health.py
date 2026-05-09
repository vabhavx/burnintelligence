"""
Heartbeat / health JSON for the intelligence pipeline. The watchdog reads this.
"""

import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HEALTH_DIR = Path.home() / ".burn_state"
HEALTH_PATH = HEALTH_DIR / "intel_health.json"

STAGES = [
    "gdelt", "bluesky", "embed", "cluster", "nvi",
    "lifecycle", "cross_narrative", "graph", "dna", "credibility",
]

_lock = threading.Lock()
_state: dict = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_stage() -> dict:
    return {"started_at": None, "completed_at": None, "status": None, "error": None}


def init(mode: str) -> None:
    """Initialize the health state for a process."""
    global _state
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _state = {
            "schema_version": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "mode": mode,
            "last_cycle_started": None,
            "last_cycle_completed": None,
            "last_cycle_duration_seconds": None,
            "last_cycle_status": None,
            "current_stage": None,
            "stage_status": {s: _empty_stage() for s in STAGES},
        }
    _flush()


def _flush() -> None:
    tmp = HEALTH_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(_state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, HEALTH_PATH)


def cycle_started() -> None:
    with _lock:
        _state["last_cycle_started"] = _now_iso()
        _state["last_cycle_completed"] = None
        _state["last_cycle_duration_seconds"] = None
        _state["last_cycle_status"] = None
        _state["stage_status"] = {s: _empty_stage() for s in STAGES}
    _flush()


def cycle_completed(duration_seconds: float, status: str) -> None:
    with _lock:
        _state["last_cycle_completed"] = _now_iso()
        _state["last_cycle_duration_seconds"] = round(duration_seconds, 1)
        _state["last_cycle_status"] = status
        _state["current_stage"] = None
    _flush()


def stage_started(stage: str) -> None:
    with _lock:
        _state["current_stage"] = stage
        _state["stage_status"][stage] = {
            "started_at": _now_iso(),
            "completed_at": None,
            "status": "running",
            "error": None,
        }
    _flush()


def stage_completed(stage: str, status: str, error: Optional[str] = None) -> None:
    with _lock:
        info = _state["stage_status"].get(stage) or _empty_stage()
        info["completed_at"] = _now_iso()
        info["status"] = status
        info["error"] = error
        _state["stage_status"][stage] = info
        if _state.get("current_stage") == stage:
            _state["current_stage"] = None
    _flush()
