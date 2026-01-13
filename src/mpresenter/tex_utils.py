from __future__ import annotations

import re


_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")
_FRAME_RE = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", re.DOTALL)
_NOTE_RE = re.compile(r"\\note(?:\[[^\]]*\])?\{")


def split_tex_document(tex: str) -> tuple[str, str, str]:
    begin_match = _BEGIN_DOC_RE.search(tex)
    end_match = _END_DOC_RE.search(tex)
    if not begin_match or not end_match or end_match.start() < begin_match.end():
        return tex, "", ""
    preamble = tex[:begin_match.end()] + "\n"
    body = tex[begin_match.end():end_match.start()]
    postamble = tex[end_match.start():]
    return preamble, body, postamble


def extract_frames(tex: str) -> list[str]:
    return _FRAME_RE.findall(tex)


def extract_notes(tex: str) -> list[str]:
    notes: list[str] = []
    index = 0
    while True:
        match = _NOTE_RE.search(tex, index)
        if not match:
            break
        start = match.end() - 1
        note_text, end = _read_braced(tex, start)
        notes.append(note_text.strip())
        index = end
    return notes


def _read_braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        return "", start
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{" and not _is_escaped(text, i):
            depth += 1
        elif ch == "}" and not _is_escaped(text, i):
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    i = index - 1
    while i >= 0 and text[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1
