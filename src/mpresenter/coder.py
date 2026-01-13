from __future__ import annotations

import base64
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .compiler import compile_latex, convert_pdf_to_pngs
from .logger import error, fix, info, thinking, warning
from .metrics import get_usage_tracker, llm_role
from .prompts import (
    coder_error_prompt,
    coder_layout_review_system_prompt,
    coder_layout_review_user_prompt,
    coder_system_prompt,
    coder_user_prompt,
)
from .tex_utils import extract_frames, extract_notes
from .utils import safe_json_loads


@dataclass
class CoderResult:
    ok: bool
    tex_code: str | None
    tex_path: Path | None
    pdf_path: Path | None

def sanitize_outline_assets(outline: Dict[str, Any], images_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    slides = outline.get("slides", [])
    for slide in slides:
        visuals = slide.get("visuals")
        if isinstance(visuals, list):
            for visual in visuals:
                if not isinstance(visual, dict):
                    continue
                image_path = visual.get("image_path")
                if not image_path:
                    continue
                image_file = images_dir / image_path
                if not image_file.exists():
                    visual["image_path"] = None
                    warnings.append(f"Missing image {image_path}; set to null")
    return outline, warnings


def _iter_image_paths(slide: Dict[str, Any]) -> List[str]:
    visuals = slide.get("visuals")
    if isinstance(visuals, list):
        paths: List[str] = []
        for visual in visuals:
            if not isinstance(visual, dict):
                continue
            image_path = visual.get("image_path")
            if image_path:
                paths.append(image_path)
        return paths
    return []


def stage_outline_images(outline: Dict[str, Any], images_dir: Path, cache_dir: Path) -> int:
    staged = 0
    slides = outline.get("slides", [])
    for slide in slides:
        for image_path in _iter_image_paths(slide):
            source = images_dir / image_path
            if not source.exists():
                continue
            target = cache_dir / image_path
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            staged += 1
    return staged


def _collect_image_metadata(outline: Dict[str, Any], images_dir: Path) -> str | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    meta: Dict[str, Dict[str, float | int | None]] = {}
    slides = outline.get("slides", [])
    for slide in slides:
        for image_path in _iter_image_paths(slide):
            if not image_path or image_path in meta:
                continue
            image_file = images_dir / image_path
            if not image_file.exists():
                continue
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception:
                continue
            aspect = round(width / height, 4) if height else None
            meta[image_path] = {"width": width, "height": height, "aspect": aspect}

    if not meta:
        return None
    return json.dumps(meta, ensure_ascii=False)


def _image_content(image_paths: List[Path]) -> List[dict]:
    content: List[dict] = []
    for path in image_paths:
        image_payload = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_payload}"})
    return content


def _labelled_image_content(prefix: str, image_paths: List[Path]) -> List[dict]:
    content: List[dict] = []
    for index, path in enumerate(image_paths, start=1):
        content.append({"type": "input_text", "text": f"{prefix} slide {index}"})
        image_payload = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_payload}"})
    return content


def _render_slide_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    slides = convert_pdf_to_pngs(pdf_path, output_dir)
    return slides


def _write_and_compile(tex_code: str, tex_path: Path, cache_dir: Path) -> tuple[bool, Path, str]:
    tex_path.write_text(tex_code, encoding="utf-8")
    success, log = compile_latex(tex_path, cache_dir)
    pdf_path = cache_dir / tex_path.with_suffix(".pdf").name
    return success, pdf_path, log


def _run_layout_review(
    llm,
    target_language: str,
    best_tex: str,
    latest_tex: str,
    best_pdf: Path,
    latest_pdf: Path,
    cache_dir: Path,
    coder_model: str,
    first_round: bool,
    last_assistant: str | None,
    item_label: str | None = None,
) -> tuple[dict[str, Any], str]:
    system_prompt = coder_layout_review_system_prompt(target_language)
    user_prompt = coder_layout_review_user_prompt(best_tex, latest_tex, target_language, first_round)

    best_slides = _render_slide_images(best_pdf, cache_dir / "slides_review_best")
    latest_slides = _render_slide_images(latest_pdf, cache_dir / "slides_review_latest")
    if not best_slides or not latest_slides:
        raise RuntimeError("No slide images available for layout review")

    prefix = f"[{item_label}] " if item_label else ""
    info("Coder", f"{prefix}Running layout review on {len(latest_slides)} slides")
    content = [{"type": "input_text", "text": user_prompt}]
    content += _labelled_image_content("BEST", best_slides)
    content += _labelled_image_content("LATEST", latest_slides)

    messages = [{"role": "system", "content": system_prompt}]
    if last_assistant:
        messages.append({"role": "assistant", "content": last_assistant})
    messages.append({"role": "user", "content": content})

    with llm_role("coder_layout"):
        response = llm.invoke_messages(messages, model=coder_model)
    payload = safe_json_loads(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Layout review returned non-JSON object")
    return payload, response


def _missing_latex_package(log: str) -> str | None:
    match = re.search(r"LaTeX Error: File `([^`]+)' not found", log)
    if match:
        return match.group(1)
    return None


def _has_overfull(log: str) -> bool:
    return re.search(r"Overfull\\s+\\\\(?:h|v)box", log, flags=re.IGNORECASE) is not None


def _extract_overfull_lines(log: str) -> str:
    lines = []
    for line in log.splitlines():
        if re.search(r"Overfull\\s+\\\\(?:h|v)box", line, flags=re.IGNORECASE):
            lines.append(line)
    return "\n".join(lines)


def _validate_notes(tex_code: str) -> tuple[bool, str]:
    frames = extract_frames(tex_code)
    notes = extract_notes(tex_code)
    if not frames:
        return False, "No frame environments found"
    if len(notes) < len(frames):
        return False, f"Missing notes: frames={len(frames)} notes={len(notes)}"
    empty = [str(i + 1) for i, note in enumerate(notes[:len(frames)]) if not note.strip()]
    if empty:
        return False, f"Empty notes on frames: {', '.join(empty)}"
    return True, ""


def _strip_metropolis_theme(tex_code: str) -> tuple[str, bool]:
    theme_patterns = [
        re.compile(r"^\s*\\usetheme\{metropolis[^}]*\}.*$"),
        re.compile(r"^\s*\\use(?:color|font|inner|outer)theme\{metropolis[^}]*\}.*$"),
        re.compile(r"^\s*\\metroset\b.*$"),
    ]
    out_lines: List[str] = []
    removed = False
    for line in tex_code.splitlines():
        if any(pattern.match(line) for pattern in theme_patterns):
            removed = True
            continue
        out_lines.append(line)
    if not removed:
        return tex_code, False
    trailing_newline = "\n" if tex_code.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline, True


def _strip_legacy_encoding_packages(tex_code: str) -> str:
    pattern = re.compile(r"^\s*\\usepackage(?:\[[^\]]*\])?\{(?:inputenc|fontenc)\}\s*$")
    out_lines: List[str] = []
    for line in tex_code.splitlines():
        if pattern.match(line):
            continue
        out_lines.append(line)
    trailing_newline = "\n" if tex_code.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline


def run_coder_loop(
    llm,
    outline: Dict[str, Any],
    target_language: str,
    cache_dir: Path,
    images_dir: Path,
    max_iters: int,
    coder_model: str,
    tex_path: Path,
    item_index: int = 1,
    total_items: int = 1,
) -> CoderResult:
    tracker = get_usage_tracker()
    item_label = f"Item {item_index}/{total_items}"
    prefix = f"[{item_label}] "
    outline, warnings = sanitize_outline_assets(outline, images_dir)
    if warnings:
        for warn_msg in warnings:
            warning("Coder", f"{prefix}{warn_msg}")
    staged = stage_outline_images(outline, images_dir, cache_dir)
    if staged:
        info("Coder", f"{prefix}Staged {staged} image assets in cache for LaTeX")

    system_prompt = coder_system_prompt(target_language)
    outline_json = json.dumps(outline, ensure_ascii=False)
    image_metadata = _collect_image_metadata(outline, images_dir)

    last_log = ""
    last_tex = ""
    last_pdf: Path | None = None
    pending_tex: str | None = None
    llm_iters = 0
    layout_max_iters = 3

    compiled_ok = False
    while llm_iters < max_iters:
        if pending_tex is None:
            llm_iters += 1
            if llm_iters == 1:
                slice_position = "FIRST" if item_index == 1 else "MIDDLE"
                user_prompt = coder_user_prompt(
                    outline_json,
                    image_metadata,
                    slice_position=slice_position,
                    item_index=item_index,
                    total_items=total_items,
                )
            else:
                prompt = coder_error_prompt("\n".join(last_log.splitlines()[-50:]))
                meta_block = f"\n\nImage Metadata (JSON):\n{image_metadata}" if image_metadata else ""
                slice_position = "FIRST" if item_index == 1 else "MIDDLE"
                user_prompt = (
                    f"{prompt}\n"
                    f"Slice Position: {slice_position} ({item_index}/{total_items})"
                    f"{meta_block}\n\n"
                    f"Previous LaTeX Code:\n{last_tex}"
                )

            thinking("Coder", f"{prefix}Generating LaTeX iteration {llm_iters}")
            with llm_role("coder_coding"):
                tex_code = llm.invoke(system_prompt, user_prompt, model=coder_model)
        else:
            tex_code = pending_tex
            pending_tex = None

        tex_code = _strip_legacy_encoding_packages(tex_code)
        last_tex = tex_code
        tex_path.write_text(tex_code, encoding="utf-8")

        success, log = compile_latex(tex_path, cache_dir)
        if not success:
            if "xelatex not found in PATH" in log:
                error("Compiler", f"{prefix}xelatex not found in PATH. Install TeX Live or add xelatex to PATH.")
                return CoderResult(False, None, None, None)

            missing_pkg = _missing_latex_package(log)
            if missing_pkg:
                if missing_pkg == "pgfopts.sty" or "metropolis" in missing_pkg:
                    fallback_tex, changed = _strip_metropolis_theme(last_tex)
                    if changed:
                        warning(
                            "Compiler",
                            f"{prefix}Missing {missing_pkg}; retrying without the metropolis theme.",
                        )
                        tex_path.write_text(fallback_tex, encoding="utf-8")
                        success, fallback_log = compile_latex(tex_path, cache_dir)
                        if success:
                            info("Compiler", f"{prefix}LaTeX compiled successfully with fallback theme")
                            last_tex = fallback_tex
                            log = fallback_log
                        else:
                            log = fallback_log
                    if not success:
                        error("Compiler", f"{prefix}Missing LaTeX package: {missing_pkg}. Install it and retry.")
                        return CoderResult(False, None, None, None)
                else:
                    error("Compiler", f"{prefix}Missing LaTeX package: {missing_pkg}. Install it and retry.")
                    return CoderResult(False, None, None, None)
            if not success:
                fix("Compiler", f"{prefix}Compilation failed. Retrying...")
                last_log = log
                continue

        if not success:
            continue

        info("Compiler", f"{prefix}LaTeX compiled successfully")
        last_pdf = cache_dir / tex_path.with_suffix(".pdf").name

        notes_ok, notes_msg = _validate_notes(last_tex)
        if not notes_ok:
            fix("Coder", f"{prefix}Notes invalid: {notes_msg}. Retrying...")
            last_log = notes_msg
            continue

        if _has_overfull(log):
            if llm_iters >= max_iters:
                overfull_log = _extract_overfull_lines(log) or log
                error(
                    "Compiler",
                    f"{prefix}Overfull boxes detected; reached max LaTeX retries.\n{overfull_log.strip()}",
                )
                return CoderResult(False, None, None, None)
            fix("Compiler", f"{prefix}Overfull boxes detected. Retrying...")
            last_log = _extract_overfull_lines(log)
            continue

        compiled_ok = True
        break

    if not compiled_ok:
        if last_log.strip():
            error(
                "Compiler",
                f"{prefix}Reached max LaTeX retries; last compile log:\n{last_log.strip()}",
            )
        else:
            error("Compiler", f"{prefix}Reached max LaTeX retries; no compile log captured")
        return CoderResult(False, None, None, None)

    if not llm.supports_vision:
        warning("Coder", f"{prefix}LLM does not support vision; skipping layout review")
        if last_pdf and last_pdf.exists():
            return CoderResult(True, last_tex, tex_path, last_pdf)
        final_ok, final_pdf, final_log = _write_and_compile(last_tex, tex_path, cache_dir)
        if not final_ok:
            error("Compiler", f"{prefix}Failed to compile final LaTeX: {final_log.strip()}")
            return CoderResult(False, None, None, None)
        return CoderResult(True, last_tex, tex_path, final_pdf)

    best_tex = last_tex
    latest_tex = last_tex
    best_tex_path = cache_dir / "slides_best.tex"
    latest_tex_path = cache_dir / "slides_latest.tex"
    last_assistant: str | None = None
    compiled_best_tex: str | None = None
    compiled_best_pdf: Path | None = None
    compiled_latest_tex: str | None = None
    compiled_latest_pdf: Path | None = None

    if tracker:
        tracker.start_span("layout_review")
    try:
        for layout_iter in range(1, layout_max_iters + 1):
            info("Coder", f"[{item_label}] Layout review iteration {layout_iter}")
            if compiled_best_tex == best_tex and compiled_best_pdf and compiled_best_pdf.exists():
                best_pdf = compiled_best_pdf
            elif last_pdf and last_pdf.exists() and best_tex == last_tex:
                best_pdf = last_pdf
                compiled_best_tex = best_tex
                compiled_best_pdf = best_pdf
            else:
                best_ok, best_pdf, best_log = _write_and_compile(best_tex, best_tex_path, cache_dir)
                if not best_ok:
                    error("Compiler", f"[{item_label}] Layout review failed to compile best version: {best_log.strip()}")
                    return CoderResult(False, None, None, None)
                compiled_best_tex = best_tex
                compiled_best_pdf = best_pdf

            if latest_tex == best_tex:
                latest_pdf = best_pdf
                compiled_latest_tex = latest_tex
                compiled_latest_pdf = latest_pdf
            elif compiled_latest_tex == latest_tex and compiled_latest_pdf and compiled_latest_pdf.exists():
                latest_pdf = compiled_latest_pdf
            else:
                latest_ok, latest_pdf, latest_log = _write_and_compile(latest_tex, latest_tex_path, cache_dir)
                if not latest_ok:
                    warning(
                        "Compiler",
                        f"[{item_label}] Layout review failed to compile latest version; "
                        f"using best version without further layout review.\n{latest_log.strip()}",
                    )
                    break
                compiled_latest_tex = latest_tex
                compiled_latest_pdf = latest_pdf

            try:
                payload, response = _run_layout_review(
                    llm,
                    target_language,
                    best_tex,
                    latest_tex,
                    best_pdf,
                    latest_pdf,
                    cache_dir,
                    coder_model,
                    first_round=(layout_iter == 1),
                    last_assistant=last_assistant,
                    item_label=item_label,
                )
            except Exception as exc:  # noqa: BLE001
                warning("Coder", f"[{item_label}] Layout review skipped: {exc}")
                break

            last_assistant = response
            status = str(payload.get("status", "FAIL")).strip().upper()
            issue_slides = payload.get("issue_slides", [])
            issues = payload.get("issues", [])
            best_version = str(payload.get("best_version", "")).strip().lower()
            comparison = payload.get("comparison")
            revised_tex = payload.get("revised_tex")

            if isinstance(revised_tex, str):
                revised_tex = _strip_legacy_encoding_packages(revised_tex)
                notes_ok, notes_msg = _validate_notes(revised_tex)
                if not notes_ok:
                    warning("Coder", f"[{item_label}] Layout review notes invalid: {notes_msg}")
                    latest_tex = revised_tex
                    continue

            if issue_slides:
                warning("Coder", f"[{item_label}] Layout issues on slides: {issue_slides}")
            if isinstance(issues, list) and issues:
                warning("Coder", f"[{item_label}] Layout issues payload: {json.dumps(issues, ensure_ascii=False)}")
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict):
                        slide = issue.get("slide")
                        problem = issue.get("problem")
                        warning("Coder", f"[{item_label}] Slide {slide}: {problem}")
                    else:
                        warning("Coder", f"[{item_label}] Layout issue: {issue}")

            if comparison:
                info("Coder", f"[{item_label}] Layout comparison: {comparison}")

            if best_version not in {"best", "latest"}:
                best_version = "latest" if layout_iter == 1 else "best"

            if status == "PASS":
                info("Coder", f"[{item_label}] Layout review passed")
                if best_version == "latest":
                    notes_ok, notes_msg = _validate_notes(latest_tex)
                    if not notes_ok:
                        warning("Coder", f"[{item_label}] Layout review notes invalid: {notes_msg}")
                        continue
                    best_tex = latest_tex
                break

            if best_version == "latest":
                best_tex = latest_tex

            if not revised_tex:
                warning("Coder", f"[{item_label}] Layout review did not return revised LaTeX; using best version")
                break

            if revised_tex is not None:
                latest_tex = revised_tex
        else:
            warning("Coder", f"[{item_label}] Reached max layout review iterations; using best version")
    finally:
        if tracker:
            tracker.end_span("layout_review")

    tex_path.write_text(best_tex, encoding="utf-8")
    final_pdf_path = cache_dir / tex_path.with_suffix(".pdf").name
    if compiled_best_tex == best_tex and compiled_best_pdf and compiled_best_pdf.exists():
        if compiled_best_pdf != final_pdf_path:
            shutil.copy2(compiled_best_pdf, final_pdf_path)
        return CoderResult(True, best_tex, tex_path, final_pdf_path)

    final_ok, final_pdf, final_log = _write_and_compile(best_tex, tex_path, cache_dir)
    if not final_ok:
        error("Compiler", f"{prefix}Failed to compile final best LaTeX: {final_log.strip()}")
        return CoderResult(False, None, None, None)
    return CoderResult(True, best_tex, tex_path, final_pdf)
