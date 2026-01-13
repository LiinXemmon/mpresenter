from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .utils import write_json


_ROLE_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("llm_role", default="unknown")
_USAGE_TRACKER: "UsageTracker | None" = None


def set_usage_tracker(tracker: Optional["UsageTracker"]) -> None:
    global _USAGE_TRACKER
    _USAGE_TRACKER = tracker


def get_usage_tracker() -> Optional["UsageTracker"]:
    return _USAGE_TRACKER


@contextlib.contextmanager
def llm_role(role: str):
    token = _ROLE_VAR.set(role)
    try:
        yield
    finally:
        _ROLE_VAR.reset(token)


def record_llm_usage(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
) -> None:
    tracker = _USAGE_TRACKER
    if tracker is None:
        return
    tracker.add_tokens(_ROLE_VAR.get(), prompt_tokens, completion_tokens, total_tokens)


@dataclass
class UsageTracker:
    cache_dir: Path
    started_at: str = field(init=False)
    durations: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    token_total: int = 0
    token_prompt_total: int = 0
    token_completion_total: int = 0
    token_by_role: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _span_counts: Dict[str, int] = field(default_factory=dict, init=False)
    _span_starts: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._start_perf = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()

    @contextlib.contextmanager
    def time_block(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.add_duration(name, elapsed)

    def start_span(self, name: str) -> None:
        with self._lock:
            count = self._span_counts.get(name, 0)
            if count == 0:
                self._span_starts[name] = time.perf_counter()
            self._span_counts[name] = count + 1

    def end_span(self, name: str) -> None:
        with self._lock:
            count = self._span_counts.get(name, 0)
            if count <= 0:
                return
            count -= 1
            if count == 0:
                start = self._span_starts.pop(name, None)
                if start is not None:
                    elapsed = time.perf_counter() - start
                    self.durations[name] = self.durations.get(name, 0.0) + elapsed
                self._span_counts.pop(name, None)
            else:
                self._span_counts[name] = count

    def add_duration(self, name: str, seconds: float) -> None:
        with self._lock:
            self.durations[name] = self.durations.get(name, 0.0) + seconds

    def set_duration(self, name: str, seconds: float) -> None:
        with self._lock:
            self.durations[name] = seconds

    def get_duration(self, name: str) -> float:
        with self._lock:
            return float(self.durations.get(name, 0.0))

    def remove_duration(self, name: str) -> None:
        with self._lock:
            self.durations.pop(name, None)

    def add_tokens(
        self,
        role: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens is None:
            return
        with self._lock:
            self.token_total += int(total_tokens)
            self.token_by_role[role] = self.token_by_role.get(role, 0) + int(total_tokens)
            if prompt_tokens is not None:
                self.token_prompt_total += int(prompt_tokens)
            if completion_tokens is not None:
                self.token_completion_total += int(completion_tokens)

    def finalize(self) -> None:
        total_elapsed = time.perf_counter() - self._start_perf
        self.set_duration("total", total_elapsed)
        with self._lock:
            now = time.perf_counter()
            for name, count in list(self._span_counts.items()):
                if count > 0:
                    start = self._span_starts.get(name)
                    if start is not None:
                        self.durations[name] = self.durations.get(name, 0.0) + (now - start)
                    self._span_counts[name] = 0
                    self._span_starts.pop(name, None)
        for key in (
            "assets_extraction",
            "outlining",
            "coding",
            "layout_review",
            "interpretation",
            "video_synthesis",
        ):
            if key not in self.durations:
                self.durations[key] = 0.0
        payload = {
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "durations_sec": dict(self.durations),
            "tokens": {
                "total": self.token_total,
                "prompt_total": self.token_prompt_total,
                "completion_total": self.token_completion_total,
                "by_role": dict(self.token_by_role),
            },
        }
        write_json(self.cache_dir / "usage_stats.json", payload)
