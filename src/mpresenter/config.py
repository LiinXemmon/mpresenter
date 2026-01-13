from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class ConfigError(RuntimeError):
    pass


def _default_session_id() -> str:
    return uuid.uuid4().hex[:8]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Config root must be an object")
    return payload


def _as_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _as_str(value: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _as_int(value: Any, key: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer") from exc


def _as_float(value: Any, key: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number") from exc


def _as_optional_float(value: Any, key: str, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be a number or null")
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number or null") from exc


def _as_bool(value: Any, key: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key} must be a boolean")


def _resolve_path(value: Any, key: str, base: Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{key} must be a path string")
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _optional_path(value: Any, key: str, base: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    return _resolve_path(value, key, base)


@dataclass
class LLMConfig:
    provider: str = "auto"
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    planner_model: str = "gpt-5-mini"
    reviewer_model: str = "gpt-5-mini"
    coder_model: str = "gpt-5-mini"
    interpreter_model: str = "gpt-5-mini"


@dataclass
class AssetConfig:
    method: str = "ppdoclayout"
    model_dir: Optional[Path] = None
    layout_unclip_ratio: Optional[float] = 1.05
    zoom: float = 3.0
    caption_max_gap: int = 180
    caption_min_x_overlap: float = 0.2
    caption_pad: int = 4
    min_area_ratio: float = 0.005
    save_layout: bool = False
    ocr_lang: Optional[str] = None
    ocr_version: Optional[str] = None


@dataclass
class TTSConfig:
    engine: str = "cosyvoice"
    cosyvoice_model: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    cosyvoice_device: str = "auto"
    cosyvoice_sample_rate: int = 22050
    cosyvoice_root: Optional[Path] = None
    cosyvoice_prompt_wav: Optional[Path] = None
    cosyvoice_prompt_text: Optional[str] = None
    cosyvoice_inference: str = "zero_shot"
    cosyvoice_speed: float = 1.0
    cosyvoice_text_frontend: bool = True
    cosyvoice_fp16: bool = False
    cosyvoice_load_trt: bool = False
    cosyvoice_load_vllm: bool = False


@dataclass
class Config:
    source_pdf: Path
    target_language: str
    session_id: str
    input_dir: Path
    cache_root: Path
    output_root: Path
    max_plan_iters: int
    max_code_iters: int
    planner_note: Optional[str]
    llm: LLMConfig
    assets: AssetConfig
    tts: TTSConfig

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / self.session_id

    @property
    def assets_manifest_path(self) -> Path:
        return self.cache_dir / "assets_manifest.json"

    @property
    def outline_path(self) -> Path:
        return self.cache_dir / "final_outline.json"

    @property
    def tex_path(self) -> Path:
        return self.cache_dir / "slides.tex"

    @property
    def final_scripts_path(self) -> Path:
        return self.cache_dir / "final_scripts.json"

    @classmethod
    def from_file(cls, project_root: Path, config_path: Path) -> "Config":
        payload = _load_json(config_path)
        return cls.from_dict(project_root, payload)

    @classmethod
    def from_dict(cls, project_root: Path, payload: dict[str, Any]) -> "Config":
        input_dir = _resolve_path(payload.get("input_dir", "input"), "input_dir", project_root)
        cache_root = _resolve_path(payload.get("cache_root", "cache"), "cache_root", project_root)
        output_root = _resolve_path(payload.get("output_root", "output"), "output_root", project_root)

        source_pdf_value = payload.get("source_pdf")
        if source_pdf_value is None:
            source_pdf = input_dir / "paper.pdf"
        else:
            source_pdf = _resolve_path(source_pdf_value, "source_pdf", project_root)

        target_language = "English"
        session_id = _default_session_id()

        max_plan_iters = _as_int(payload.get("max_plan_iters"), "max_plan_iters", 3)
        max_code_iters = _as_int(payload.get("max_code_iters"), "max_code_iters", 6)
        planner_note = _as_str(payload.get("planner_note"), "planner_note")
        if planner_note is not None:
            planner_note = planner_note.strip() or None

        llm_data = _as_section(payload, "llm")
        provider = (_as_str(llm_data.get("provider"), "llm.provider", "auto") or "auto").strip().lower()
        api_key = _as_str(llm_data.get("api_key"), "llm.api_key")
        openai_api_key = _as_str(llm_data.get("openai_api_key"), "llm.openai_api_key")
        gemini_api_key = _as_str(llm_data.get("gemini_api_key"), "llm.gemini_api_key")
        if api_key is not None:
            api_key = api_key.strip() or None
        if openai_api_key is not None:
            openai_api_key = openai_api_key.strip() or None
        if gemini_api_key is not None:
            gemini_api_key = gemini_api_key.strip() or None
        if openai_api_key is None:
            openai_api_key = api_key
        llm = LLMConfig(
            provider=provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            planner_model=_as_str(llm_data.get("planner_model"), "llm.planner_model", "gpt-5-mini") or "gpt-5-mini",
            reviewer_model=_as_str(llm_data.get("reviewer_model"), "llm.reviewer_model", "gpt-5-mini") or "gpt-5-mini",
            coder_model=_as_str(llm_data.get("coder_model"), "llm.coder_model", "gpt-5-mini") or "gpt-5-mini",
            interpreter_model=_as_str(
                llm_data.get("interpreter_model"),
                "llm.interpreter_model",
                "gpt-5-mini",
            )
            or "gpt-5-mini",
        )

        assets_data = _as_section(payload, "assets")
        asset_method = (_as_str(assets_data.get("method"), "assets.method", "ppdoclayout") or "ppdoclayout").strip()
        if "layout_unclip_ratio" in assets_data:
            layout_unclip_ratio = _as_optional_float(
                assets_data.get("layout_unclip_ratio"),
                "assets.layout_unclip_ratio",
                None,
            )
        else:
            layout_unclip_ratio = 1.05

        assets = AssetConfig(
            method=asset_method.lower(),
            model_dir=_optional_path(assets_data.get("model_dir"), "assets.model_dir", project_root),
            layout_unclip_ratio=layout_unclip_ratio,
            zoom=_as_float(assets_data.get("zoom"), "assets.zoom", 3.0),
            caption_max_gap=_as_int(assets_data.get("caption_max_gap"), "assets.caption_max_gap", 180),
            caption_min_x_overlap=_as_float(
                assets_data.get("caption_min_x_overlap"),
                "assets.caption_min_x_overlap",
                0.2,
            ),
            caption_pad=_as_int(assets_data.get("caption_pad"), "assets.caption_pad", 4),
            min_area_ratio=_as_float(assets_data.get("min_area_ratio"), "assets.min_area_ratio", 0.005),
            save_layout=_as_bool(assets_data.get("save_layout"), "assets.save_layout", False),
            ocr_lang=_as_str(assets_data.get("ocr_lang"), "assets.ocr_lang"),
            ocr_version=_as_str(assets_data.get("ocr_version"), "assets.ocr_version"),
        )

        tts_data = _as_section(payload, "tts")
        tts_engine = (_as_str(tts_data.get("engine"), "tts.engine", "cosyvoice") or "cosyvoice").strip().lower()
        if tts_engine in {"cosyvoice3", "cosyvoice-3"}:
            tts_engine = "cosyvoice"
        if "cosyvoice" in tts_engine:
            tts_engine = "cosyvoice"
        cosyvoice_model = (
            _as_str(
                tts_data.get("cosyvoice_model"),
                "tts.cosyvoice_model",
                "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            )
            or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        )
        cosyvoice_device = (_as_str(tts_data.get("cosyvoice_device"), "tts.cosyvoice_device", "auto") or "auto")
        cosyvoice_sample_rate = _as_int(tts_data.get("cosyvoice_sample_rate"), "tts.cosyvoice_sample_rate", 22050)
        cosyvoice_root = _optional_path(tts_data.get("cosyvoice_root"), "tts.cosyvoice_root", project_root)
        cosyvoice_prompt_wav = _optional_path(
            tts_data.get("cosyvoice_prompt_wav"),
            "tts.cosyvoice_prompt_wav",
            project_root,
        )
        cosyvoice_prompt_text = _as_str(tts_data.get("cosyvoice_prompt_text"), "tts.cosyvoice_prompt_text")
        cosyvoice_inference = (
            _as_str(tts_data.get("cosyvoice_inference"), "tts.cosyvoice_inference", "zero_shot") or "zero_shot"
        ).strip().lower()
        cosyvoice_speed = _as_float(tts_data.get("cosyvoice_speed"), "tts.cosyvoice_speed", 1.0)
        cosyvoice_text_frontend = _as_bool(
            tts_data.get("cosyvoice_text_frontend"),
            "tts.cosyvoice_text_frontend",
            True,
        )
        cosyvoice_fp16 = _as_bool(tts_data.get("cosyvoice_fp16"), "tts.cosyvoice_fp16", False)
        cosyvoice_load_trt = _as_bool(tts_data.get("cosyvoice_load_trt"), "tts.cosyvoice_load_trt", False)
        cosyvoice_load_vllm = _as_bool(tts_data.get("cosyvoice_load_vllm"), "tts.cosyvoice_load_vllm", False)

        tts = TTSConfig(
            engine=tts_engine,
            cosyvoice_model=cosyvoice_model,
            cosyvoice_device=cosyvoice_device,
            cosyvoice_sample_rate=cosyvoice_sample_rate,
            cosyvoice_root=cosyvoice_root,
            cosyvoice_prompt_wav=cosyvoice_prompt_wav,
            cosyvoice_prompt_text=cosyvoice_prompt_text,
            cosyvoice_inference=cosyvoice_inference,
            cosyvoice_speed=cosyvoice_speed,
            cosyvoice_text_frontend=cosyvoice_text_frontend,
            cosyvoice_fp16=cosyvoice_fp16,
            cosyvoice_load_trt=cosyvoice_load_trt,
            cosyvoice_load_vllm=cosyvoice_load_vllm,
        )

        return cls(
            source_pdf=source_pdf,
            target_language=target_language,
            session_id=session_id,
            input_dir=input_dir,
            cache_root=cache_root,
            output_root=output_root,
            max_plan_iters=max_plan_iters,
            max_code_iters=max_code_iters,
            planner_note=planner_note,
            llm=llm,
            assets=assets,
            tts=tts,
        )
