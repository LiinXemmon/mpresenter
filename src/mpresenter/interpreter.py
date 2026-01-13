from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .logger import info, vision, warning
from .metrics import llm_role
from .prompts import interpreter_system_prompt, interpreter_user_prompt
from .utils import safe_json_loads, write_json


def run_interpreter_from_manifest(
    llm,
    slides_manifest: List[Dict[str, Any]],
    target_language: str,
    cache_dir: Path,
    model: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    system_prompt = interpreter_system_prompt(target_language)
    final_scripts: List[Dict[str, Any]] = []
    has_all_images = True

    for entry in slides_manifest:
        note = entry.get("note") or ""
        image_path_value = entry.get("image_path")
        image_path = Path(image_path_value) if image_path_value else None
        if image_path and not image_path.exists():
            has_all_images = False
            image_path = None

        final_script = note
        modified = False

        if image_path and llm.supports_vision:
            user_prompt = interpreter_user_prompt(note, target_language)
            with llm_role("interpreter"):
                response_text = llm.invoke_vision(system_prompt, user_prompt, model=model, image_path=image_path)
            try:
                payload = safe_json_loads(response_text)
                modified = bool(payload.get("modified", False))
                final_script = payload.get("final_script", note)
            except Exception:
                final_script = note

        slide_id = entry.get("slide_id") or entry.get("outline_id") or entry.get("slide_index")
        if note:
            ratio = len(final_script) / len(note)
            if ratio > 1.5:
                vision("Interpreter", f"Script expanded by {ratio:.2f}x for slide {slide_id}")

        final_scripts.append(
            {
                "id": slide_id,
                "original": note,
                "final": final_script,
                "modified": modified,
            }
        )

    write_json(cache_dir / "final_scripts.json", final_scripts)
    info("Interpreter", f"Processed {len(final_scripts)} scripts")
    if not has_all_images and slides_manifest:
        warning("Interpreter", "Slide images missing; scripts generated without visual grounding")
    return final_scripts, has_all_images


def run_interpreter(llm, slides_dir: Path, outline: Dict[str, Any], target_language: str,
                    cache_dir: Path, model: str) -> Tuple[List[Dict[str, Any]], bool]:
    slides = outline.get("slides", [])
    slide_images = sorted(slides_dir.glob("slide-*.png"))
    has_all_images = len(slides) == 0 or len(slide_images) >= len(slides)
    system_prompt = interpreter_system_prompt(target_language)

    final_scripts: List[Dict[str, Any]] = []

    for index, slide in enumerate(slides):
        original_script = slide.get("speech_script", "")
        image_path = slide_images[index] if index < len(slide_images) else None

        final_script = original_script
        modified = False

        if image_path and llm.supports_vision:
            user_prompt = interpreter_user_prompt(original_script, target_language)
            with llm_role("interpreter"):
                response_text = llm.invoke_vision(system_prompt, user_prompt, model=model, image_path=image_path)
            try:
                payload = safe_json_loads(response_text)
                modified = bool(payload.get("modified", False))
                final_script = payload.get("final_script", original_script)
            except Exception:
                final_script = original_script

        if len(original_script) > 0:
            ratio = len(final_script) / len(original_script)
            if ratio > 1.5:
                vision("Interpreter", f"Script expanded by {ratio:.2f}x for slide {slide.get('id')}")

        final_scripts.append({
            "id": slide.get("id"),
            "original": original_script,
            "final": final_script,
            "modified": modified,
        })

    write_json(cache_dir / "final_scripts.json", final_scripts)
    info("Interpreter", f"Processed {len(final_scripts)} scripts")
    if not has_all_images and slides:
        warning("Interpreter", "Slide images missing; scripts generated without visual grounding")
    return final_scripts, has_all_images
