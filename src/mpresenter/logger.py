from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
from typing import Optional


_log_path: Path | None = None
_log_lock = threading.Lock()


def set_log_path(path: Path | None) -> None:
    global _log_path
    _log_path = path


def _emit(level: str, component: str, message: str, color: Optional[str] = None) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{level}] [{component}]"
    line = f"{timestamp} {prefix} {message}"
    print(line, flush=True)
    if _log_path is None:
        return
    try:
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            with _log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        # Avoid recursive logging if file writes fail.
        pass


def info(component: str, message: str) -> None:
    _emit("INFO", component, message)


def thinking(component: str, message: str) -> None:
    _emit("Thinking", component, message)


def critique(component: str, message: str) -> None:
    _emit("Critique", component, message)


def error(component: str, message: str) -> None:
    _emit("Error", component, message)


def warning(component: str, message: str) -> None:
    _emit("Warning", component, message)


def fix(component: str, message: str) -> None:
    _emit("Fix", component, message)


def vision(component: str, message: str) -> None:
    _emit("Vision", component, message)
