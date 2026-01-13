from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Iterable, List, Optional

from .logger import error, info
from .utils import ensure_dir


SLIDE_PAUSE_SECONDS = 0.5


def synthesize_video(
    slides: Iterable[Path],
    audio_files: Iterable[Path],
    output_path: Path,
    work_dir: Optional[Path] = None,
) -> bool:
    if shutil.which("ffmpeg") is None:
        error("Synthesizer", "ffmpeg not found in PATH")
        return False

    slides = list(slides)
    audio_files = list(audio_files)
    if not slides:
        error("Synthesizer", "No slide images available")
        return False

    if work_dir is None:
        work_dir = output_path.parent
    ensure_dir(work_dir)

    segments_dir = work_dir / "segments"
    ensure_dir(segments_dir)
    segments: List[Path] = []

    for index, slide_path in enumerate(slides):
        audio_path = audio_files[index] if index < len(audio_files) else None
        if audio_path is None:
            error("Synthesizer", f"Missing audio for slide {index + 1}")
            return False

        segment_path = segments_dir / f"segment_{index:02d}.mkv"
        duration = _audio_duration_seconds(audio_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(slide_path),
            "-i",
            str(audio_path),
        ]
        if duration:
            segment_duration = duration + SLIDE_PAUSE_SECONDS
            cmd.extend(["-t", f"{segment_duration:.6f}"])
            if SLIDE_PAUSE_SECONDS > 0:
                cmd.extend(["-af", f"apad=pad_dur={SLIDE_PAUSE_SECONDS:.3f}"])
        cmd += [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(segment_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error("Synthesizer", f"ffmpeg segment failed: {result.stderr.strip()}")
            return False
        segments.append(segment_path)

    concat_list = work_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for segment in segments:
            try:
                path_text = segment.relative_to(work_dir).as_posix()
            except ValueError:
                path_text = segment.resolve().as_posix()
            handle.write(f"file '{path_text}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error("Synthesizer", f"ffmpeg concat failed: {result.stderr.strip()}")
        return False

    info("Synthesizer", f"Final video written to {output_path}")
    return True


def _audio_duration_seconds(audio_path: Path) -> float | None:
    if not audio_path.exists():
        return None
    try:
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except Exception:
        return None
    if rate <= 0:
        return None
    return frames / float(rate)
