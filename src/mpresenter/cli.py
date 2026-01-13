from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import Config, ConfigError
from .llm import GeminiClient, MultiLLMClient, OpenAIClient, RetryingLLMClient
from .logger import error, info, set_log_path
from .pipeline import run_pipeline


def build_llm(config: Config):
    provider = (config.llm.provider or "").strip().lower()
    openai_key = config.llm.openai_api_key
    gemini_key = config.llm.gemini_api_key

    def _build_openai() -> OpenAIClient | None:
        if not openai_key:
            return None
        return OpenAIClient(api_key=openai_key)

    def _build_gemini() -> GeminiClient | None:
        if gemini_key:
            return GeminiClient(api_key=gemini_key)
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return GeminiClient(api_key=None)
        return None

    if provider == "openai":
        client = _build_openai()
        if not client:
            raise RuntimeError("OpenAI provider requires llm.openai_api_key in config.json")
        info("System", "Using OpenAI LLM client")
        return RetryingLLMClient(client, max_attempts=5)
    if provider in {"gemini", "google"}:
        client = _build_gemini()
        if not client:
            raise RuntimeError("Gemini provider requires llm.gemini_api_key in config.json")
        info("System", "Using Gemini LLM client")
        return RetryingLLMClient(client, max_attempts=5)
    if provider in {"auto", "mixed", "multi", ""}:
        clients = {}
        openai_client = _build_openai()
        gemini_client = _build_gemini()
        if openai_client:
            clients["openai"] = openai_client
        if gemini_client:
            clients["gemini"] = gemini_client
        if not clients:
            raise RuntimeError("Auto provider requires llm.openai_api_key and/or llm.gemini_api_key in config.json")
        default_provider = "openai" if "openai" in clients else next(iter(clients.keys()))
        info("System", f"Using multi-provider LLM client (default={default_provider})")
        return RetryingLLMClient(MultiLLMClient(clients, default_provider=default_provider), max_attempts=5)

    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-to-Video Agentic System")
    parser.add_argument("--source-pdf", type=str, help="Path to the input PDF")
    parser.add_argument("--output-root", type=str, help="Path to the output directory")
    parser.add_argument("--target-language", type=str, help="Target language for slides/scripts (default: English)")
    parser.add_argument("--cache-root", type=str, help="Path to the cache root directory")
    parser.add_argument("--planner-note", type=str, help="Supplementary metadata note for the planner")
    parser.add_argument(
        "--backbone",
        type=str,
        help="Override all LLM models (planner/reviewer/coder/interpreter) with a single model name",
    )
    return parser.parse_args()

def _resolve_cli_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = parse_args()
    try:
        config = Config.from_file(project_root, project_root / "config.json")
    except ConfigError as exc:
        error("System", str(exc))
        return

    source_pdf = _resolve_cli_path(args.source_pdf, project_root)
    output_root = _resolve_cli_path(args.output_root, project_root)
    cache_root = _resolve_cli_path(args.cache_root, project_root)

    if source_pdf:
        config.source_pdf = source_pdf
    if output_root:
        config.output_root = output_root
    if cache_root:
        config.cache_root = cache_root
    if args.planner_note is not None:
        note = args.planner_note.strip()
        config.planner_note = note or None
    language = (args.target_language or "English").strip() or "English"
    config.target_language = language
    if args.backbone:
        backbone = args.backbone.strip()
        if backbone:
            config.llm.planner_model = backbone
            config.llm.reviewer_model = backbone
            config.llm.coder_model = backbone
            config.llm.interpreter_model = backbone
    session_id = config.source_pdf.stem.strip()
    if not session_id:
        session_id = "session"
    config.session_id = session_id
    set_log_path(config.cache_dir / "run.log")
    if config.planner_note:
        info("System", "Planner note:")
        for line in config.planner_note.splitlines():
            info("System", f"  {line}")
    else:
        info("System", "Planner note: (none)")

    try:
        llm = build_llm(config)
        run_pipeline(config, llm)
    except RuntimeError as exc:
        error("System", str(exc))


if __name__ == "__main__":
    main()
