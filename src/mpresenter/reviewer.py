from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .logger import critique
from .metrics import llm_role
from .prompts import reviewer_system_prompt, reviewer_user_prompt
from .utils import safe_json_loads, write_json


def review_outline(
    llm,
    outline: Dict[str, Any],
    target_language: str,
    cache_dir: Path,
    iteration: int,
    model: str,
    planner_explanation: Any = None,
    planner_note: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[Dict[str, Any], list[dict[str, Any]]]:
    outline_json = json.dumps(outline, ensure_ascii=False)
    system_prompt = reviewer_system_prompt(target_language)
    explanation_json = None
    if planner_explanation is not None:
        explanation_json = json.dumps(planner_explanation, ensure_ascii=False)
    user_prompt = reviewer_user_prompt(target_language, outline_json, explanation_json, planner_note)

    if messages is None:
        messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    with llm_role("reviewer"):
        response_text = llm.invoke_messages(messages, model=model)
    messages.append({"role": "assistant", "content": response_text})

    feedback = safe_json_loads(response_text)
    write_json(cache_dir / f"feedback_v{iteration}.json", feedback)

    status = feedback.get("status", "FAIL")
    critique("Reviewer", f"{status}: {feedback.get('critique', '')}")
    return feedback, messages
