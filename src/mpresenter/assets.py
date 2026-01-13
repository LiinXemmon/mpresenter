from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AssetConfig
from .logger import error, info, vision, warning
from .utils import ensure_dir, write_json


@dataclass
class CandidateBox:
    page: int
    bbox_px: Tuple[int, int, int, int]
    label: str
    score_hint: float = 0.0


@dataclass
class TitleBox:
    page: int
    kind: Optional[str]
    label: str
    bbox_px: Tuple[int, int, int, int]
    score: Optional[float] = None


CAPTION_SEP = r"[:.\s\-\uFF1A\uFF0E\u3001]*"
FIG_PATTERNS = [
    re.compile(rf"^\s*(Figure|Fig\.?)\s*([0-9\uff10-\uff19]+){CAPTION_SEP}", re.IGNORECASE),
    re.compile(rf"^\s*\u56fe\s*([0-9\uff10-\uff19]+){CAPTION_SEP}"),
]
TAB_PATTERNS = [
    re.compile(rf"^\s*(Table)\s*([0-9\uff10-\uff19]+){CAPTION_SEP}", re.IGNORECASE),
    re.compile(rf"^\s*\u8868\s*([0-9\uff10-\uff19]+){CAPTION_SEP}"),
]

CN_NUM = {
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10,
}
CN_FIG = re.compile(rf"^\s*\u56fe\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+){CAPTION_SEP}")
CN_TAB = re.compile(rf"^\s*\u8868\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+){CAPTION_SEP}")

_FULLWIDTH_DIGITS = str.maketrans("\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19", "0123456789")


def normalize_digits(text: str) -> str:
    if not text:
        return text
    text = text.translate(_FULLWIDTH_DIGITS)
    out = []
    for ch in text:
        if ch.isdigit() and not ("0" <= ch <= "9"):
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


def cn_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    if text in CN_NUM:
        return CN_NUM[text]
    if text.startswith("\u5341"):
        tail = text[1:]
        return 10 + (CN_NUM.get(tail, 0) if tail else 0)
    if "\u5341" in text:
        left, right = text.split("\u5341", 1)
        return CN_NUM.get(left, 0) * 10 + (CN_NUM.get(right, 0) if right else 0)
    return None


def _match_caption_kind(line: str) -> Optional[Tuple[str, Optional[int]]]:
    text = line.strip()
    for pat in FIG_PATTERNS:
        match = pat.match(text)
        if match:
            number = None
            for group in match.groups():
                if group and group.isdigit():
                    number = int(normalize_digits(group))
            return "figure", number
    for pat in TAB_PATTERNS:
        match = pat.match(text)
        if match:
            number = None
            for group in match.groups():
                if group and group.isdigit():
                    number = int(normalize_digits(group))
            return "table", number

    match = CN_FIG.match(text)
    if match:
        return "figure", cn_to_int(match.group(1))
    match = CN_TAB.match(text)
    if match:
        return "table", cn_to_int(match.group(1))

    return None


def parse_caption_from_text(text: str) -> Optional[Tuple[str, int]]:
    if not text:
        return None
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    for line in lines:
        match = _match_caption_kind(line)
        if match and match[1] is not None:
            return match[0], match[1]
    match = _match_caption_kind(text.strip())
    if match and match[1] is not None:
        return match[0], match[1]
    return None


def _default_ocr_lang(target_language: str) -> str:
    normalized = (target_language or "").strip().lower()
    if normalized in {"chinese", "zh", "zh-cn", "zh-hans", "zh-hant", "cn"}:
        return "ch"
    return "en"


def _candidate_kind_from_label(label: str) -> Optional[str]:
    if label == "table":
        return "table"
    if label in {"image", "chart"}:
        return "figure"
    return None


def _title_kind_from_label(label: str) -> Optional[str]:
    if label == "table_title":
        return "table"
    if label in {"figure_title", "chart_title"}:
        return "figure"
    return None


def _vertical_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if by1 < ay0:
        return ay0 - by1
    if ay1 < by0:
        return by0 - ay1
    return 0.0


def x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    denom = max(1, min(ax1 - ax0, bx1 - bx0))
    return inter / denom


def infer_caption_direction(
    candidates: List[CandidateBox],
    titles: List[TitleBox],
    max_gap: float,
    min_overlap: float,
    dir_margin: float = 6.0,
) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {
        "figure": {"above": 0, "below": 0},
        "table": {"above": 0, "below": 0},
    }
    for cand in candidates:
        cand_kind = _candidate_kind_from_label(cand.label)
        if cand_kind is None:
            continue
        best_score = None
        best_title = None
        for title in titles:
            if title.kind != cand_kind:
                continue
            gap = _vertical_gap(cand.bbox_px, title.bbox_px)
            if gap > max_gap:
                continue
            overlap = x_overlap_ratio(cand.bbox_px, title.bbox_px)
            if overlap < min_overlap:
                continue
            score = gap + (1.0 - overlap) * 10.0
            if best_score is None or score < best_score:
                best_score = score
                best_title = title
        if best_title is None:
            continue
        if best_title.bbox_px[1] >= cand.bbox_px[3] - dir_margin:
            counts[cand_kind]["below"] += 1
        elif best_title.bbox_px[3] <= cand.bbox_px[1] + dir_margin:
            counts[cand_kind]["above"] += 1

    prefer: Dict[str, str] = {}
    for kind, tally in counts.items():
        above = tally["above"]
        below = tally["below"]
        if above == 0 and below == 0:
            prefer[kind] = "none"
        elif above > below:
            prefer[kind] = "above"
        elif below > above:
            prefer[kind] = "below"
        else:
            prefer[kind] = "none"
    return prefer


def match_candidates_to_titles(
    candidates: List[CandidateBox],
    titles: List[TitleBox],
    max_gap: float,
    min_overlap: float,
    prefer_dir: Optional[Dict[str, str]] = None,
    dir_margin: float = 6.0,
) -> Dict[int, int]:
    pairs: List[Tuple[float, int, int]] = []
    for ci, cand in enumerate(candidates):
        for ti, title in enumerate(titles):
            gap = _vertical_gap(cand.bbox_px, title.bbox_px)
            if gap > max_gap:
                continue
            overlap = x_overlap_ratio(cand.bbox_px, title.bbox_px)
            if overlap < min_overlap:
                continue
            cand_kind = _candidate_kind_from_label(cand.label) or title.kind or ""
            title_kind = title.kind or ""
            kind_pen = 30.0 if cand_kind and title_kind and cand_kind != title_kind else 0.0
            dir_pen = 0.0
            if cand_kind:
                prefer = (prefer_dir or {}).get(cand_kind, "none")
                if prefer == "below":
                    if title.bbox_px[1] >= cand.bbox_px[3] - dir_margin:
                        dir_pen = 0.0
                    elif title.bbox_px[3] <= cand.bbox_px[1] + dir_margin:
                        dir_pen = 200.0
                    else:
                        dir_pen = 120.0
                elif prefer == "above":
                    if title.bbox_px[3] <= cand.bbox_px[1] + dir_margin:
                        dir_pen = 0.0
                    elif title.bbox_px[1] >= cand.bbox_px[3] - dir_margin:
                        dir_pen = 200.0
                    else:
                        dir_pen = 120.0
            score = gap + kind_pen + dir_pen + (1.0 - overlap) * 10.0 + 10.0 * cand.score_hint
            pairs.append((score, ci, ti))

    pairs.sort(key=lambda x: x[0])
    matched: Dict[int, int] = {}
    used_cands: set[int] = set()
    used_titles: set[int] = set()
    for _, ci, ti in pairs:
        if ci in used_cands or ti in used_titles:
            continue
        used_cands.add(ci)
        used_titles.add(ti)
        matched[ci] = ti
    return matched


def detect_candidates_ppdoclayout(
    model,
    page_img,
    page_index: int,
    return_raw: bool = False,
) -> Tuple[List[CandidateBox], List[TitleBox], Optional[List[Dict[str, object]]]]:
    import numpy as np

    img_arr = np.array(page_img)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    img_arr = img_arr[:, :, ::-1].copy()
    layout = model.predict(img_arr)

    candidates: List[CandidateBox] = []
    titles: List[TitleBox] = []
    raw_boxes: List[Dict[str, object]] = []
    candidate_labels = {"image", "chart", "table"}
    title_labels = {"figure_title", "table_title", "chart_title"}
    for item in layout or []:
        boxes = item.get("boxes") or []
        for box in boxes:
            label = str(box.get("label", "")).lower()
            raw_boxes.append(box)
            coord = box.get("coordinate")
            if not coord or len(coord) != 4:
                continue
            x1, y1, x2, y2 = map(int, coord)
            score = box.get("score")
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                score_val = None
            if label in candidate_labels:
                score_hint = 0.0 if score_val is None else (1.0 - score_val)
                candidates.append(
                    CandidateBox(
                        page=page_index,
                        bbox_px=(x1, y1, x2, y2),
                        label=label,
                        score_hint=score_hint,
                    )
                )
            if label in title_labels:
                titles.append(
                    TitleBox(
                        page=page_index,
                        kind=_title_kind_from_label(label),
                        label=label,
                        bbox_px=(x1, y1, x2, y2),
                        score=score_val,
                    )
                )
    return candidates, titles, (raw_boxes if return_raw else None)


def ocr_caption_text(
    ocr_model,
    page_img,
    bbox: Tuple[int, int, int, int],
    pad: int = 4,
) -> str:
    import numpy as np

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(page_img.width, x1 + pad)
    y1 = min(page_img.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = page_img.crop((x0, y0, x1, y1))
    img_arr = np.array(crop)
    crop.close()
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    img_arr = img_arr[:, :, ::-1].copy()
    if not hasattr(ocr_model, "predict"):
        warning("Extractor", "PaddleOCR model does not support predict(); caption OCR skipped")
        return ""
    res = ocr_model.predict(
        img_arr,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if not res:
        return ""
    rec_texts = res[0].get("rec_texts", []) if isinstance(res[0], dict) else []
    parts = [t.strip() for t in rec_texts if isinstance(t, str) and t.strip()]
    return " ".join(parts)


def log_drop(
    page: int,
    label: str,
    bbox: Tuple[int, int, int, int],
    reason: str,
    extra: Optional[str] = None,
) -> None:
    msg = f"[DROP] page={page} label={label} bbox_px={bbox} reason={reason}"
    if extra:
        msg = f"{msg} {extra}"
    info("Extractor", msg)


def _load_ppdoclayout_model(model_dir: Optional[str], layout_unclip_ratio: Optional[float]):
    from paddleocr import LayoutDetection  # type: ignore

    kwargs: Dict[str, object] = {"model_name": "PP-DocLayout-L"}
    if model_dir:
        kwargs["model_dir"] = model_dir
    if layout_unclip_ratio is not None:
        kwargs["layout_unclip_ratio"] = layout_unclip_ratio
    return LayoutDetection(**kwargs)


def _load_ocr_model(lang: Optional[str], ocr_version: Optional[str]):
    from paddleocr import PaddleOCR  # type: ignore

    kwargs: Dict[str, object] = {}
    if lang:
        kwargs["lang"] = lang
    if ocr_version:
        kwargs["ocr_version"] = ocr_version
    return PaddleOCR(**kwargs)


def _render_page_image(page, zoom: float):
    import fitz  # type: ignore
    from PIL import Image

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_assets(
    source_pdf: Path,
    output_root: Path,
    cache_dir: Path,
    target_language: str,
    asset_config: AssetConfig,
    *,
    images_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    if images_dir is None:
        images_dir = output_root / "images"
    ensure_dir(images_dir)

    manifest: List[Dict[str, Any]] = []

    if not source_pdf.exists():
        error("Extractor", f"Source PDF not found: {source_pdf}")
        write_json(cache_dir / "assets_manifest.json", manifest)
        return manifest, False

    method = (asset_config.method or "").strip().lower()
    if method == "ppdoclayout":
        pp_manifest = _extract_assets_ppdoclayout(
            source_pdf,
            images_dir,
            cache_dir,
            target_language,
            asset_config,
        )
        if pp_manifest is not None:
            write_json(cache_dir / "assets_manifest.json", pp_manifest)
            info("Extractor", f"Extracted {len(pp_manifest)} assets via PP-DocLayout-L")
            return pp_manifest, True

        warning("Extractor", "PP-DocLayout-L extraction failed; no assets extracted")
        write_json(cache_dir / "assets_manifest.json", manifest)
        return manifest, False

    error("Extractor", f"Unsupported asset extraction method: {method}")
    write_json(cache_dir / "assets_manifest.json", manifest)
    return manifest, False


def _extract_assets_ppdoclayout(
    source_pdf: Path,
    images_dir: Path,
    cache_dir: Path,
    target_language: str,
    asset_config: AssetConfig,
) -> Optional[List[Dict[str, Any]]]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        error("Extractor", f"PP-DocLayout-L requires PyMuPDF (fitz): {exc}")
        return None

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        error("Extractor", f"PP-DocLayout-L requires numpy: {exc}")
        return None

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        error("Extractor", f"PP-DocLayout-L requires Pillow: {exc}")
        return None

    try:
        from paddleocr import LayoutDetection, PaddleOCR  # type: ignore
    except Exception:
        error("Extractor", "PP-DocLayout-L requires paddleocr + paddlepaddle")
        return None

    model_dir = str(asset_config.model_dir) if asset_config.model_dir else None
    layout_unclip_ratio = asset_config.layout_unclip_ratio
    zoom = asset_config.zoom
    caption_max_gap = asset_config.caption_max_gap
    caption_min_x_overlap = asset_config.caption_min_x_overlap
    caption_pad = asset_config.caption_pad
    min_area_ratio = asset_config.min_area_ratio
    save_layout = asset_config.save_layout
    ocr_lang = asset_config.ocr_lang or _default_ocr_lang(target_language)
    ocr_version = asset_config.ocr_version

    try:
        doc = fitz.open(source_pdf)
    except Exception as exc:
        error("Extractor", f"Failed to open PDF with PyMuPDF: {exc}")
        return None

    try:
        layout_model = _load_ppdoclayout_model(model_dir, layout_unclip_ratio)
    except Exception as exc:
        error("Extractor", f"Failed to load PP-DocLayout-L model: {exc}")
        return None

    try:
        ocr_model = _load_ocr_model(ocr_lang, ocr_version)
    except Exception as exc:
        error("Extractor", f"Failed to load PaddleOCR model: {exc}")
        return None

    per_page: List[Dict[str, object]] = []
    layout_debug: List[Dict[str, object]] = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        page_img = _render_page_image(page, zoom)

        candidates, titles, raw_layout = detect_candidates_ppdoclayout(
            layout_model,
            page_img,
            pno,
            return_raw=save_layout,
        )
        if save_layout and raw_layout is not None:
            for box in raw_layout:
                coord = box.get("coordinate")
                if coord is not None:
                    coord = [float(x) for x in coord]
                score = box.get("score")
                try:
                    score = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score = None
                layout_debug.append(
                    {
                        "page": pno + 1,
                        "label": box.get("label"),
                        "score": score,
                        "bbox_px": coord,
                    }
                )

        page_area = float(page_img.width * page_img.height)
        filtered: List[CandidateBox] = []
        for cand in candidates:
            x0, y0, x1, y1 = cand.bbox_px
            if x1 <= x0 or y1 <= y0:
                log_drop(pno + 1, cand.label, cand.bbox_px, "invalid_bbox")
                continue
            area = max(1, (x1 - x0) * (y1 - y0))
            area_ratio = area / page_area
            if area_ratio >= min_area_ratio:
                filtered.append(cand)
            else:
                log_drop(
                    pno + 1,
                    cand.label,
                    cand.bbox_px,
                    "area_ratio_below_threshold",
                    f"ratio={area_ratio:.4f} min={min_area_ratio:.4f}",
                )

        per_page.append(
            {
                "page": pno,
                "candidates": filtered,
                "titles": titles,
            }
        )
        page_img.close()

    all_candidates: List[CandidateBox] = []
    all_titles: List[TitleBox] = []
    for entry in per_page:
        all_candidates.extend(entry["candidates"])
        all_titles.extend(entry["titles"])
    global_prefer = infer_caption_direction(
        all_candidates,
        all_titles,
        max_gap=caption_max_gap,
        min_overlap=caption_min_x_overlap,
    )

    items_with_order: List[Tuple[Tuple[int, float, float], Dict[str, Any]]] = []
    used_files: set[str] = set()
    for entry in per_page:
        pno = entry["page"]
        candidates = entry["candidates"]
        titles = entry["titles"]

        page = doc.load_page(pno)
        page_img = _render_page_image(page, zoom)

        matches = match_candidates_to_titles(
            candidates,
            titles,
            max_gap=caption_max_gap,
            min_overlap=caption_min_x_overlap,
            prefer_dir=global_prefer,
        )

        matched_candidates = set(matches.keys())
        for ci, cand in enumerate(candidates):
            if ci not in matched_candidates:
                log_drop(pno + 1, cand.label, cand.bbox_px, "no_title_match")

        page_items: List[Dict[str, Any]] = []
        for ci, ti in matches.items():
            cand = candidates[ci]
            title = titles[ti]
            caption_text = ocr_caption_text(
                ocr_model,
                page_img,
                title.bbox_px,
                pad=caption_pad,
            )
            if not caption_text.strip():
                log_drop(pno + 1, cand.label, cand.bbox_px, "ocr_empty")
                continue
            parsed = parse_caption_from_text(caption_text)
            if not parsed:
                log_drop(
                    pno + 1,
                    cand.label,
                    cand.bbox_px,
                    "caption_parse_failed",
                    f"text={caption_text!r}",
                )
                continue
            kind, number = parsed
            file_name = f"Figure_{number}.png" if kind == "figure" else f"Table_{number}.png"
            if file_name in used_files:
                log_drop(
                    pno + 1,
                    cand.label,
                    cand.bbox_px,
                    "duplicate_caption_number",
                    f"name={file_name}",
                )
                continue
            used_files.add(file_name)

            bbox_list = [float(x) for x in cand.bbox_px]
            crop = _crop_image(page_img, bbox_list)
            if crop is None:
                error(
                    "Extractor",
                    f"Invalid bbox on page {pno + 1} for {file_name}: {bbox_list}",
                )
                continue

            output_path = images_dir / file_name
            crop.save(output_path)
            crop.close()
            vision("Extractor", f"Saved {file_name} from page {pno + 1}")

            item = {
                "name": file_name,
                "type": kind,
                "caption": caption_text.strip(),
                "page": pno + 1,
                "bbox": bbox_list,
            }
            order_key = (pno, float(title.bbox_px[1]), float(title.bbox_px[0]))
            items_with_order.append((order_key, item))
            page_items.append(item)

        write_json(cache_dir / f"assets_page_{pno + 1:03d}.json", {"items": page_items})
        page_img.close()

    if save_layout:
        write_json(cache_dir / "assets_layout.json", layout_debug)

    items_with_order.sort(key=lambda entry: entry[0])
    return [item for _, item in items_with_order]


def _normalize_bbox(bbox: List[float], width: int, height: int) -> Optional[List[int]]:
    if max(bbox) <= 1.0:
        x1, y1, x2, y2 = bbox
        bbox = [x1 * width, y1 * height, x2 * width, y2 * height]

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, width - 1))
    x2 = max(1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(1, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return [x1, y1, x2, y2]


def _crop_image(image, bbox: List[float]):
    width, height = image.size
    normalized = _normalize_bbox(bbox, width, height)
    if normalized is None:
        return None
    x1, y1, x2, y2 = normalized
    return image.crop((x1, y1, x2, y2))
