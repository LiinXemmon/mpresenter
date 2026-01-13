from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import re

from .logger import error, info, warning
from .utils import ensure_dir


def compile_latex(tex_path: Path, work_dir: Path) -> Tuple[bool, str]:
    if shutil.which("xelatex") is None:
        return False, "xelatex not found in PATH"

    env = os.environ.copy()

    tex_name = tex_path.name
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(work_dir),
        tex_name,
    ]
    # info("Compiler", f"Running compile command: {' '.join(cmd)} (cwd={tex_path.parent})")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tex_path.parent)
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    pdf_path = work_dir / tex_path.with_suffix(".pdf").name

    if result.returncode != 0 or not pdf_path.exists():
        return False, log

    return True, log


def convert_pdf_to_pngs(pdf_path: Path, output_dir: Path, *, dpi: int = 150, zoom: float = 2.0) -> List[Path]:
    ensure_dir(output_dir)
    for stale in output_dir.glob("slide-*.png"):
        stale.unlink(missing_ok=True)

    if shutil.which("pdftoppm"):
        prefix = output_dir / "slide"
        cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error("Compiler", f"pdftoppm failed: {result.stderr.strip()}")
            return []
        slides = _normalize_slide_names(output_dir)
        info("Compiler", f"Rendered {len(slides)} slides using pdftoppm")
        return slides

    if shutil.which("convert"):
        output_pattern = output_dir / "slide-%02d.png"
        cmd = ["convert", "-density", str(dpi), str(pdf_path), str(output_pattern)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error("Compiler", f"convert failed: {result.stderr.strip()}")
            return []
        slides = _normalize_slide_names(output_dir)
        info("Compiler", f"Rendered {len(slides)} slides using ImageMagick")
        return slides

    try:
        import fitz  # type: ignore
    except Exception as exc:
        error("Compiler", f"No PDF renderer found and PyMuPDF unavailable: {exc}")
        return []

    try:
        with fitz.open(pdf_path) as doc:
            slides: List[Path] = []
            matrix = fitz.Matrix(zoom, zoom)
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                output_path = output_dir / f"slide-{index:03d}.png"
                pix.save(str(output_path))
                slides.append(output_path)
    except Exception as exc:
        error("Compiler", f"PyMuPDF rendering failed: {exc}")
        return []

    info("Compiler", f"Rendered {len(slides)} slides using PyMuPDF")
    return slides


def _normalize_slide_names(output_dir: Path) -> List[Path]:
    slides = list(output_dir.glob("slide-*.png"))

    def _index(path: Path) -> int:
        match = re.search(r"slide-(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    slides_sorted = sorted(slides, key=_index)
    normalized: List[Path] = []
    for index, slide_path in enumerate(slides_sorted, start=1):
        target = output_dir / f"slide-{index:03d}.png"
        if slide_path != target:
            slide_path.replace(target)
        normalized.append(target)
    return normalized
