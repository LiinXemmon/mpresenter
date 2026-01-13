from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

from .utils import read_json, write_json


@dataclass
class State:
    session_id: str
    steps: Dict[str, bool] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, session_id: str, path: Path, steps: Iterable[str]) -> "State":
        if path.exists():
            data = read_json(path)
            return cls(session_id=session_id, steps=data.get("steps", {}), path=path)

        initial = {step: False for step in steps}
        state = cls(session_id=session_id, steps=initial, path=path)
        state.save()
        return state

    def save(self) -> None:
        if not self.path:
            return
        payload = {"session_id": self.session_id, "steps": self.steps}
        write_json(self.path, payload)

    def is_done(self, step: str) -> bool:
        return self.steps.get(step, False)

    def mark_done(self, step: str) -> None:
        self.steps[step] = True
        self.save()
