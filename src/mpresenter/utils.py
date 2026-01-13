from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def safe_json_loads(text: str) -> Any:
    raw = text.strip()
    first_exc: json.JSONDecodeError | None = None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        first_exc = exc

    fenced = _extract_fenced_json(raw)
    if fenced is not None:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            repaired = _escape_invalid_backslashes(fenced)
            if repaired != fenced:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    parsed = _scan_for_json(raw)
    if parsed is not None:
        return parsed

    repaired_raw = _escape_invalid_backslashes(raw)
    if repaired_raw != raw:
        try:
            return json.loads(repaired_raw)
        except json.JSONDecodeError:
            parsed = _scan_for_json(repaired_raw)
            if parsed is not None:
                return parsed

    _log_json_decode_error(raw)
    if first_exc is not None:
        raise first_exc
    raise json.JSONDecodeError("Expecting value", raw, 0)


def _extract_fenced_json(text: str) -> str | None:
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _scan_for_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch in "{[":
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
    return None


def _log_json_decode_error(text: str) -> None:
    try:
        from .logger import error
    except Exception:
        print("JSON decode failed. Raw text:")
        print(text)
        return
    error("Parser", f"JSON decode failed. Raw text:\n{text}")


def _escape_invalid_backslashes(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    for index, ch in enumerate(text):
        if not in_string:
            if ch == "\"":
                in_string = True
            out.append(ch)
            continue
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            nxt = text[index + 1] if index + 1 < len(text) else ""
            if nxt in "\"\\/bfnrt":
                out.append(ch)
                escape = True
                continue
            if nxt == "u" and index + 5 < len(text):
                hex_part = text[index + 2 : index + 6]
                if all(c in string.hexdigits for c in hex_part):
                    out.append(ch)
                    escape = True
                    continue
            out.append("\\\\")
            continue
        if ch == "\"":
            in_string = False
        out.append(ch)
    return "".join(out)


def sanitize_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    return escaped


def chunk_text(text: str, delimiters: Iterable[str] = (".", "!", "?", ";", ":")) -> list[str]:
    if not text:
        return []
    pattern = "|".join(re.escape(d) for d in delimiters)
    parts = re.split(f"({pattern})", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if part in delimiters:
            chunk = buffer.strip()
            if chunk:
                chunks.append(chunk)
            buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks
