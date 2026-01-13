from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .llm import Attachment
from .logger import info, thinking
from .metrics import llm_role
from .prompts import planner_system_prompt, planner_user_prompt
from .reviewer import review_outline
from .utils import safe_json_loads, write_json


def run_planner_loop(
    llm,
    source_pdf: Path,
    target_language: str,
    assets_manifest: List[Any],
    cache_dir: Path,
    max_iters: int,
    planner_model: str,
    reviewer_model: str,
    planner_note: str | None = None,
) -> Dict[str, Any]:
    assets_manifest_list = _format_assets_manifest(assets_manifest)
    system_prompt = planner_system_prompt(target_language, assets_manifest_list)
    attachments = [Attachment(kind="file", path=source_pdf)]

    outline: Dict[str, Any] = {}
    critique_text = None
    questions = None
    planner_explanation = None
    planner_messages = [{"role": "system", "content": system_prompt}]
    reviewer_messages = None

    for iteration in range(1, max_iters + 1):
        if iteration == 1:
            user_prompt = planner_user_prompt(target_language, assets_manifest_list, planner_note=planner_note)
        else:
            user_prompt = planner_user_prompt(
                target_language,
                assets_manifest_list,
                critique=critique_text,
                questions=questions,
                planner_note=planner_note,
            )

        thinking("Planner", f"Generating outline iteration {iteration}")
        planner_messages.append({"role": "user", "content": user_prompt})
        with llm_role("planner"):
            response_text = llm.invoke_messages(
                planner_messages,
                model=planner_model,
                attachments=attachments if iteration == 1 else None,
            )
        planner_messages.append({"role": "assistant", "content": response_text})
        payload = safe_json_loads(response_text)
        outline, planner_explanation = _split_outline_payload(payload)
        if iteration == 1:
            planner_explanation = None
        write_json(cache_dir / f"outline_v{iteration}.json", outline)
        if planner_explanation is not None:
            write_json(cache_dir / f"planner_explanation_v{iteration}.json", planner_explanation)

        feedback, reviewer_messages = review_outline(
            llm,
            outline,
            target_language,
            cache_dir,
            iteration,
            reviewer_model,
            planner_explanation,
            planner_note,
            reviewer_messages,
        )
        if feedback.get("status") == "PASS":
            info("Planner", "Outline accepted by reviewer")
            break

        critique_text = feedback.get("critique", "")
        questions = feedback.get("questions", feedback.get("expert_questions", []))

    write_json(cache_dir / "final_outline.json", outline)
    return outline


def _format_assets_manifest(assets_manifest: List[Any]) -> str:
    if not assets_manifest:
        return "None"
    if all(isinstance(item, str) for item in assets_manifest):
        return ", ".join(assets_manifest)

    lines = []
    for item in assets_manifest:
        if isinstance(item, dict):
            name = item.get("name") or item.get("filename") or "unknown"
            caption = item.get("caption")
            asset_type = item.get("type")
            parts = [str(name)]
            if asset_type:
                parts.append(f"({asset_type})")
            if caption:
                parts.append(f": {caption}")
            lines.append(" ".join(parts).strip())
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _split_outline_payload(payload: Any) -> tuple[Dict[str, Any], Any]:
    if not isinstance(payload, dict):
        return {"slides": []}, None
    if "outline" in payload and isinstance(payload["outline"], dict):
        outline = payload["outline"]
        explanation = payload.get("planner_explanation")
        return outline, explanation
    outline = dict(payload)
    explanation = outline.pop("planner_explanation", None)
    return outline, explanation
