from __future__ import annotations

import concurrent.futures
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .assets import extract_assets
from .coder import run_coder_loop
from .compiler import compile_latex, convert_pdf_to_pngs
from .interpreter import run_interpreter_from_manifest
from .logger import error, fix, info, set_log_path, warning
from .metrics import UsageTracker, llm_role, set_usage_tracker
from .planner import run_planner_loop
from .prompts import (
    tex_fix_system_prompt,
    tex_fix_user_prompt,
    tex_merge_system_prompt,
    tex_merge_user_prompt,
)
from .state import State
from .synthesizer import synthesize_audio
from .tex_utils import extract_notes
from .utils import ensure_dir, read_json, write_json
from .video import synthesize_video

STEPS = [
    "EXTRACT_ASSETS",
    "PLAN_OUTLINE",
    "GENERATE_CODE",
    "INTERPRET_VISUALS",
    "SYNTHESIZE_VIDEO",
]

LLM_DPI = 150
LLM_ZOOM = 2.0
VIDEO_DPI = 300
VIDEO_ZOOM = 3.0


def _prepare_output_slides(output_dir: Path) -> None:
    ensure_dir(output_dir)
    for stale in output_dir.glob("slide-*.png"):
        stale.unlink(missing_ok=True)


def _missing_latex_package(log: str) -> str | None:
    match = re.search(r"LaTeX Error: File `([^`]+)' not found", log)
    if match:
        return match.group(1)
    return None


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


def _combine_tex_documents(
    llm,
    tex_paths: List[Path],
    target_language: str,
    model: str,
) -> Optional[str]:
    if not tex_paths:
        return None
    if len(tex_paths) == 1:
        return tex_paths[0].read_text(encoding="utf-8")

    tex_docs = [path.read_text(encoding="utf-8") for path in tex_paths]
    system_prompt = tex_merge_system_prompt(target_language)
    user_prompt = tex_merge_user_prompt(tex_docs)
    with llm_role("coder_coding"):
        merged = llm.invoke(system_prompt, user_prompt, model=model)
    if not merged.strip():
        error("Compiler", "LLM merge returned empty LaTeX")
        return None
    return merged


def _compile_combined_tex(
    llm,
    tex_code: str,
    tex_path: Path,
    cache_dir: Path,
    target_language: str,
    model: str,
    max_iters: int,
) -> Tuple[bool, str, Optional[Path]]:
    last_log = ""
    for attempt in range(1, max_iters + 1):
        tex_path.write_text(tex_code, encoding="utf-8")
        success, log = compile_latex(tex_path, cache_dir)
        if success:
            info("Compiler", f"Combined LaTeX compiled successfully (attempt {attempt})")
            pdf_path = cache_dir / tex_path.with_suffix(".pdf").name
            return True, tex_code, pdf_path

        if "xelatex not found in PATH" in log:
            error("Compiler", "xelatex not found in PATH. Install TeX Live or add xelatex to PATH.")
            return False, tex_code, None

        missing_pkg = _missing_latex_package(log)
        if missing_pkg:
            if missing_pkg == "pgfopts.sty" or "metropolis" in missing_pkg:
                fallback_tex, changed = _strip_metropolis_theme(tex_code)
                if changed:
                    warning(
                        "Compiler",
                        f"Missing {missing_pkg}; retrying combined TeX without the metropolis theme.",
                    )
                    tex_code = fallback_tex
                    continue
            error("Compiler", f"Missing LaTeX package: {missing_pkg}. Install it and retry.")
            return False, tex_code, None

        last_log = log
        fix("Compiler", f"Combined LaTeX failed (attempt {attempt}); requesting minimal fixes")
        error_log_snippet = "\n".join(last_log.splitlines()[-50:])
        system_prompt = tex_fix_system_prompt(target_language)
        user_prompt = tex_fix_user_prompt(error_log_snippet, tex_code)
        with llm_role("coder_coding"):
            tex_code = llm.invoke(system_prompt, user_prompt, model=model)

    if last_log.strip():
        error("Compiler", f"Reached max combined LaTeX retries; last compile log:\n{last_log.strip()}")
    else:
        error("Compiler", "Reached max combined LaTeX retries; no compile log captured")
    return False, tex_code, None


def _make_slide_id(outline_id: Any, local_index: int, local_total: int) -> str:
    base = str(outline_id) if outline_id is not None else str(local_index)
    if local_total > 1:
        return f"{base}-{local_index}"
    return base


def _process_outline_item(
    llm,
    slide: Dict[str, Any],
    item_index: int,
    total: int,
    target_language: str,
    cache_root: Path,
    images_dir: Path,
    max_code_iters: int,
    coder_model: str,
) -> Dict[str, Any]:
    item_dir = cache_root / f"item_{item_index:03d}"
    ensure_dir(item_dir)
    item_tex_path = item_dir / "slides.tex"
    item_outline = {"slides": [slide]}

    info("Coder", f"[Item {item_index}/{total}] start")
    coder_result = run_coder_loop(
        llm,
        item_outline,
        target_language,
        item_dir,
        images_dir,
        max_code_iters,
        coder_model,
        item_tex_path,
        item_index,
        total,
    )

    if not coder_result.ok or not coder_result.tex_path or not coder_result.pdf_path:
        raise RuntimeError("LaTeX generation failed")

    pdf_path = coder_result.pdf_path
    if not pdf_path.exists():
        raise RuntimeError(f"Missing PDF: {pdf_path}")

    item_slides_dir = item_dir / "slides"
    slide_images = convert_pdf_to_pngs(pdf_path, item_slides_dir)
    if not slide_images:
        raise RuntimeError("No slide images generated")

    tex_code = coder_result.tex_code or coder_result.tex_path.read_text(encoding="utf-8")
    notes = extract_notes(tex_code)
    if len(notes) < len(slide_images):
        raise RuntimeError(f"Missing notes: {len(notes)} notes vs {len(slide_images)} slides")

    outline_id = slide.get("id") if isinstance(slide, dict) else item_index
    info("Coder", f"[Item {item_index}/{total}] done ({len(slide_images)} slides)")

    return {
        "item_index": item_index,
        "outline_id": outline_id,
        "tex_path": coder_result.tex_path,
        "pdf_path": pdf_path,
        "slide_images": slide_images,
        "notes": notes,
    }


def run_pipeline(config, llm) -> None:
    set_log_path(config.cache_dir / "run.log")
    tracker = UsageTracker(config.cache_dir)
    set_usage_tracker(tracker)
    ensure_dir(config.cache_dir)
    state_path = config.cache_dir / "steps_status.json"
    state = State.load(config.session_id, state_path, STEPS)

    try:
        info("System", f"Session ID: {config.session_id}. Source PDF: {config.source_pdf}")

        if not config.source_pdf.exists():
            error("System", f"Source PDF not found. Place a file at {config.source_pdf} or pass --source-pdf")
            return

        if not state.is_done("EXTRACT_ASSETS"):
            with tracker.time_block("assets_extraction"):
                assets_manifest, assets_ok = extract_assets(
                    config.source_pdf,
                    config.output_root,
                    config.cache_dir,
                    config.target_language,
                    config.assets,
                    images_dir=config.cache_dir,
                )
            if assets_ok:
                state.mark_done("EXTRACT_ASSETS")
        else:
            assets_manifest = read_json(config.assets_manifest_path)

        if not state.is_done("PLAN_OUTLINE"):
            with tracker.time_block("outlining"):
                outline = run_planner_loop(
                    llm,
                    config.source_pdf,
                    config.target_language,
                    assets_manifest,
                    config.cache_dir,
                    config.max_plan_iters,
                    config.llm.planner_model,
                    config.llm.reviewer_model,
                    config.planner_note,
                )
            state.mark_done("PLAN_OUTLINE")
        else:
            outline = read_json(config.outline_path)

        slides_manifest: List[Dict[str, Any]]
        manifest_path = config.cache_dir / "slides_manifest.json"

        if not state.is_done("GENERATE_CODE"):
            with tracker.time_block("coding_total"):
                slides = outline.get("slides", [])
                if not isinstance(slides, list) or not slides:
                    error("Coder", "Outline contains no slides to render")
                    return

                llm_slides_dir = config.cache_dir / "slides_llm"
                video_slides_dir = config.cache_dir / "slides"
                _prepare_output_slides(llm_slides_dir)
                _prepare_output_slides(video_slides_dir)

                slides_manifest = []
                item_tex_paths: List[Path] = []
                global_index = 1

                max_workers = min(10, len(slides))
                info("Coder", f"Processing {len(slides)} outline items with {max_workers} workers")
                results: Dict[int, Dict[str, Any]] = {}
                failures: List[int] = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            _process_outline_item,
                            llm,
                            slide,
                            item_index,
                            len(slides),
                            config.target_language,
                            config.cache_dir,
                            config.cache_dir,
                            config.max_code_iters,
                            config.llm.coder_model,
                        ): item_index
                        for item_index, slide in enumerate(slides, start=1)
                    }

                    for future in concurrent.futures.as_completed(future_map):
                        item_index = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            error("Coder", f"[Item {item_index}] failed: {exc}")
                            failures.append(item_index)
                        else:
                            results[item_index] = result

                if failures:
                    error("Coder", f"{len(failures)} outline items failed; aborting pipeline")
                    return

                for item_index in sorted(results.keys()):
                    result = results[item_index]
                    outline_id = result["outline_id"]
                    slide_images = result["slide_images"]
                    notes = result["notes"]
                    pdf_path = result["pdf_path"]
                    tex_path = result["tex_path"]
                    item_tex_paths.append(Path(tex_path))

                    local_total = len(slide_images)
                    for local_index, slide_image in enumerate(slide_images, start=1):
                        slide_id = _make_slide_id(outline_id, local_index, local_total)
                        output_path = llm_slides_dir / f"slide-{global_index:03d}.png"
                        shutil.copy2(slide_image, output_path)
                        slides_manifest.append(
                            {
                                "slide_index": global_index,
                                "slide_id": slide_id,
                                "outline_index": item_index,
                                "outline_id": outline_id,
                                "local_index": local_index,
                                "image_path": str(output_path),
                                "note": notes[local_index - 1],
                                "tex_path": str(tex_path),
                                "pdf_path": str(pdf_path),
                            }
                        )
                        global_index += 1

                write_json(manifest_path, slides_manifest)

                combined_tex = _combine_tex_documents(
                    llm,
                    item_tex_paths,
                    config.target_language,
                    config.llm.coder_model,
                )
                if not combined_tex:
                    error("Compiler", "Failed to assemble combined LaTeX document")
                    return

                combined_ok, combined_tex, combined_pdf = _compile_combined_tex(
                    llm,
                    combined_tex,
                    config.tex_path,
                    config.cache_dir,
                    config.target_language,
                    config.llm.coder_model,
                    config.max_code_iters,
                )
                if not combined_ok:
                    return
                if not combined_pdf or not combined_pdf.exists():
                    error("Compiler", "Combined PDF missing after merge")
                    return

                merged_slides = convert_pdf_to_pngs(
                    combined_pdf,
                    llm_slides_dir,
                    dpi=LLM_DPI,
                    zoom=LLM_ZOOM,
                )
                if not merged_slides:
                    error("Compiler", "Failed to render combined slides")
                    return
                if len(merged_slides) != len(slides_manifest):
                    error(
                        "Compiler",
                        f"Combined slide count mismatch: merged={len(merged_slides)} "
                        f"manifest={len(slides_manifest)}",
                    )
                    return
                for entry, merged_path in zip(slides_manifest, merged_slides):
                    entry["image_path"] = str(merged_path)
                write_json(manifest_path, slides_manifest)

                video_slides = convert_pdf_to_pngs(
                    combined_pdf,
                    video_slides_dir,
                    dpi=VIDEO_DPI,
                    zoom=VIDEO_ZOOM,
                )
                if not video_slides:
                    error("Compiler", "Failed to render high-resolution slides for video")
                    return

                state.mark_done("GENERATE_CODE")

            layout_time = tracker.get_duration("layout_review")
            coding_total = tracker.get_duration("coding_total")
            tracker.set_duration("coding", max(coding_total - layout_time, 0.0))
            tracker.remove_duration("coding_total")
        else:
            slides_manifest = read_json(manifest_path)
            llm_slides_dir = config.cache_dir / "slides_llm"
            video_slides_dir = config.cache_dir / "slides"
            combined_pdf = config.cache_dir / config.tex_path.with_suffix(".pdf").name
            if combined_pdf.exists():
                merged_slides = convert_pdf_to_pngs(
                    combined_pdf,
                    llm_slides_dir,
                    dpi=LLM_DPI,
                    zoom=LLM_ZOOM,
                )
                if merged_slides and isinstance(slides_manifest, list):
                    if len(merged_slides) != len(slides_manifest):
                        error(
                            "Compiler",
                            f"Combined slide count mismatch: merged={len(merged_slides)} "
                            f"manifest={len(slides_manifest)}",
                        )
                        return
                    for entry, merged_path in zip(slides_manifest, merged_slides):
                        entry["image_path"] = str(merged_path)
                    write_json(manifest_path, slides_manifest)

                video_slides = convert_pdf_to_pngs(
                    combined_pdf,
                    video_slides_dir,
                    dpi=VIDEO_DPI,
                    zoom=VIDEO_ZOOM,
                )
                if not video_slides:
                    error("Compiler", "Failed to render high-resolution slides for video")
                    return
            else:
                warning("Compiler", "Combined PDF missing; using existing slide images")

        if not state.is_done("INTERPRET_VISUALS"):
            if not isinstance(slides_manifest, list) or not slides_manifest:
                error("Interpreter", "Slides manifest missing or invalid")
                return
            with tracker.time_block("interpretation"):
                _, interpreter_ok = run_interpreter_from_manifest(
                    llm,
                    slides_manifest,
                    config.target_language,
                    config.cache_dir,
                    config.llm.interpreter_model,
                )
            if interpreter_ok:
                state.mark_done("INTERPRET_VISUALS")

        if not state.is_done("SYNTHESIZE_VIDEO"):
            with tracker.time_block("video_synthesis"):
                scripts = read_json(config.final_scripts_path)
                audio_paths = synthesize_audio(scripts, config.cache_dir / "audio", config.target_language, config.tts)
                slide_images = sorted((config.cache_dir / "slides").glob("slide-*.png"))
                output_video = config.output_root / f"{config.source_pdf.stem}.mp4"
                ensure_dir(config.output_root)
                video_ok = synthesize_video(slide_images, audio_paths, output_video, work_dir=config.cache_dir)
            if video_ok:
                state.mark_done("SYNTHESIZE_VIDEO")
    finally:
        tracker.finalize()
        set_usage_tracker(None)
