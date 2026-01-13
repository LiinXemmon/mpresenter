from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Iterable, Optional

import httpx

from .logger import warning
from .metrics import record_llm_usage


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_openai_usage(response) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None, None

    if isinstance(usage, dict):
        prompt_tokens = _coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        completion_tokens = _coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_tokens = _coerce_int(usage.get("total_tokens"))
        return prompt_tokens, completion_tokens, total_tokens

    prompt_tokens = _coerce_int(getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None))
    completion_tokens = _coerce_int(
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    )
    total_tokens = _coerce_int(getattr(usage, "total_tokens", None))
    return prompt_tokens, completion_tokens, total_tokens


def _extract_gemini_usage(response: dict) -> tuple[int | None, int | None, int | None]:
    usage = response.get("usageMetadata", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return None, None, None
    prompt_tokens = _coerce_int(usage.get("promptTokenCount"))
    completion_tokens = _coerce_int(usage.get("candidatesTokenCount"))
    total_tokens = _coerce_int(usage.get("totalTokenCount"))
    return prompt_tokens, completion_tokens, total_tokens

@dataclass
class Attachment:
    kind: str
    path: Path


class LLMClient:
    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def invoke_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def invoke_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        image_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    @property
    def supports_vision(self) -> bool:
        return False


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAIClient") from exc

        self._client = OpenAI(api_key=api_key)
        self._file_cache: dict[Path, str] = {}

    @property
    def supports_vision(self) -> bool:
        return True

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.invoke_messages(
            messages,
            model=model,
            attachments=attachments,
            reasoning_effort=reasoning_effort,
        )

    def invoke_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        if attachments:
            self._inject_attachments(messages, attachments)
        payload = {
            "model": model,
            "input": messages,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        response = self._client.responses.create(**payload)
        prompt_tokens, completion_tokens, total_tokens = _extract_openai_usage(response)
        record_llm_usage(prompt_tokens, completion_tokens, total_tokens)
        return response.output_text

    def invoke_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        image_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        image_payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_payload}"},
        ]
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        response = self._client.responses.create(**payload)
        prompt_tokens, completion_tokens, total_tokens = _extract_openai_usage(response)
        record_llm_usage(prompt_tokens, completion_tokens, total_tokens)
        return response.output_text

    def _ensure_file_id(self, path: Path) -> str:
        resolved = path.resolve()
        cached = self._file_cache.get(resolved)
        if cached:
            return cached
        with resolved.open("rb") as handle:
            uploaded = self._client.files.create(file=handle, purpose="user_data")
        file_id = uploaded.id
        self._file_cache[resolved] = file_id
        return file_id

    def _inject_attachments(self, messages: list[dict], attachments: Iterable[Attachment]) -> None:
        if not attachments:
            return
        user_index = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                user_index = idx
                break
        if user_index is None:
            return

        user_msg = messages[user_index]
        content = user_msg.get("content", "")
        file_items = []
        for attachment in attachments:
            file_id = self._ensure_file_id(attachment.path)
            file_items.append({
                "type": "input_file",
                "file_id": file_id,
            })

        if isinstance(content, list):
            user_msg["content"] = file_items + content
        else:
            user_msg["content"] = file_items + [{"type": "input_text", "text": str(content)}]


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "Gemini provider requires GEMINI_API_KEY/GOOGLE_API_KEY or llm.gemini_api_key in config.json"
            )
        self._base_url = (base_url or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    @property
    def supports_vision(self) -> bool:
        return True

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.invoke_messages(
            messages,
            model=model,
            attachments=attachments,
            reasoning_effort=reasoning_effort,
        )

    def invoke_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        system_prompt = self._system_prompt_from_messages(messages)
        contents = self._messages_to_contents(messages)
        if attachments:
            contents = self._append_attachments(contents, attachments)

        payload = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            response = self._request(model, payload)
        except RuntimeError as exc:
            if system_prompt and "systemInstruction" in str(exc):
                payload.pop("systemInstruction", None)
                payload["contents"] = self._inject_system_as_text(system_prompt, contents)
                response = self._request(model, payload)
            else:
                raise

        prompt_tokens, completion_tokens, total_tokens = _extract_gemini_usage(response)
        record_llm_usage(prompt_tokens, completion_tokens, total_tokens)
        return self._extract_response_text(response)

    def invoke_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        image_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{image_payload}"
        content = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": data_url},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self.invoke_messages(messages, model=model, reasoning_effort=reasoning_effort)

    def _system_prompt_from_messages(self, messages: list[dict]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return self._extract_text_from_content(msg.get("content", ""))
        return ""

    def _messages_to_contents(self, messages: list[dict]) -> list[dict]:
        contents: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            parts = self._content_to_parts(msg.get("content", ""))
            if not parts:
                continue
            contents.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": parts,
                }
            )
        return contents

    def _content_to_parts(self, content) -> list[dict]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return [{"text": str(content)}]

        parts: list[dict] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "input_text":
                    parts.append({"text": str(item.get("text", ""))})
                    continue
                if item_type == "input_image":
                    part = self._data_url_to_part(str(item.get("image_url", "")))
                    if part:
                        parts.append(part)
                    continue
                if item_type == "input_file":
                    continue
            parts.append({"text": str(item)})
        return parts

    def _data_url_to_part(self, data_url: str) -> Optional[dict]:
        if not data_url.startswith("data:") or ";base64," not in data_url:
            return None
        header, b64_data = data_url.split(",", 1)
        mime_type = header.split(":", 1)[1].split(";", 1)[0]
        return {"inline_data": {"mime_type": mime_type, "data": b64_data}}

    def _append_attachments(self, contents: list[dict], attachments: Iterable[Attachment]) -> list[dict]:
        parts: list[dict] = []
        for attachment in attachments:
            path = attachment.path
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})

        if not parts:
            return contents

        for content in contents:
            if content.get("role") == "user":
                content["parts"].extend(parts)
                return contents

        return contents + [{"role": "user", "parts": parts}]

    def _inject_system_as_text(self, system_prompt: str, contents: list[dict]) -> list[dict]:
        if not contents:
            return [{"role": "user", "parts": [{"text": system_prompt}]}]
        if contents[0].get("role") != "user":
            return [{"role": "user", "parts": [{"text": system_prompt}]}] + contents
        contents[0]["parts"].insert(0, {"text": system_prompt + "\n\n"})
        return contents

    def _request(self, model: str, payload: dict) -> dict:
        url = f"{self._base_url}/models/{model}:generateContent"
        response = httpx.post(url, params={"key": self._api_key}, json=payload, timeout=120.0)
        if response.status_code >= 400:
            message = self._error_message(response)
            raise RuntimeError(f"Gemini API error {response.status_code}: {message}")
        return response.json()

    def _error_message(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except Exception:
            return response.text.strip()
        if isinstance(data, dict):
            error = data.get("error", {})
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
        return str(data)

    def _extract_response_text(self, response: dict) -> str:
        candidates = response.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini response missing candidates")
        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part.get("text")))
        return "".join(texts).strip()

    def _extract_text_from_content(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts)
        return str(content)


def _split_provider_prefix(model: str) -> tuple[str | None, str]:
    if ":" not in model:
        return None, model
    prefix, rest = model.split(":", 1)
    prefix = prefix.strip().lower()
    if prefix in {"openai", "gemini", "google"}:
        provider = "gemini" if prefix == "google" else prefix
        return provider, rest.strip()
    return None, model


def _infer_provider_from_model(model: str) -> str | None:
    name = (model or "").strip().lower()
    if not name:
        return None
    if name.startswith(("gemini", "models/gemini", "google/gemini")) or "gemini" in name:
        return "gemini"
    if name.startswith(("gpt", "o1", "o3", "o4", "openai")) or "gpt-" in name:
        return "openai"
    return None


class MultiLLMClient(LLMClient):
    def __init__(self, clients: dict[str, LLMClient], default_provider: str | None = None) -> None:
        if not clients:
            raise RuntimeError("MultiLLMClient requires at least one provider client")
        self._clients = clients
        if default_provider in clients:
            self._default_provider = default_provider
        else:
            self._default_provider = next(iter(clients.keys()))

    @property
    def supports_vision(self) -> bool:
        return any(client.supports_vision for client in self._clients.values())

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        client, resolved_model = self._select_client(model)
        return client.invoke(
            system_prompt,
            user_prompt,
            model=resolved_model,
            attachments=attachments,
            reasoning_effort=reasoning_effort,
        )

    def invoke_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        client, resolved_model = self._select_client(model)
        return client.invoke_messages(
            messages,
            model=resolved_model,
            attachments=attachments,
            reasoning_effort=reasoning_effort,
        )

    def invoke_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        image_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        client, resolved_model = self._select_client(model)
        return client.invoke_vision(
            system_prompt,
            user_prompt,
            model=resolved_model,
            image_path=image_path,
            reasoning_effort=reasoning_effort,
        )

    def _select_client(self, model: str) -> tuple[LLMClient, str]:
        provider, cleaned_model = _split_provider_prefix(model)
        if provider:
            client = self._clients.get(provider)
            if client is None:
                raise RuntimeError(f"Provider {provider} not configured")
            return client, cleaned_model

        inferred = _infer_provider_from_model(model)
        if inferred:
            client = self._clients.get(inferred)
            if client is None:
                raise RuntimeError(f"Model {model} requires provider {inferred}, but it is not configured")
            return client, model

        return self._clients[self._default_provider], model


class RetryingLLMClient(LLMClient):
    def __init__(self, client: LLMClient, max_attempts: int = 5, base_delay: float = 1.0,
                 max_delay: float = 8.0) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay

    @property
    def supports_vision(self) -> bool:
        return self._client.supports_vision

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        return self._with_retry(
            lambda: self._client.invoke(
                system_prompt,
                user_prompt,
                model=model,
                attachments=attachments,
                reasoning_effort=reasoning_effort,
            ),
            "invoke",
        )

    def invoke_messages(
        self,
        messages: list[dict],
        *,
        model: str,
        attachments: Optional[Iterable[Attachment]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        return self._with_retry(
            lambda: self._client.invoke_messages(
                messages,
                model=model,
                attachments=attachments,
                reasoning_effort=reasoning_effort,
            ),
            "invoke_messages",
        )

    def invoke_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        image_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        return self._with_retry(
            lambda: self._client.invoke_vision(
                system_prompt,
                user_prompt,
                model=model,
                image_path=image_path,
                reasoning_effort=reasoning_effort,
            ),
            "invoke_vision",
        )

    def _with_retry(self, fn, label: str):
        last_exc: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if not self._should_retry(exc, attempt):
                    raise
                last_exc = exc
                delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
                jitter = random.uniform(0, delay * 0.1)
                sleep_for = delay + jitter
                warning(
                    "LLM",
                    f"{label} failed (attempt {attempt}/{self._max_attempts}): {exc}. "
                    f"Retrying in {sleep_for:.1f}s",
                )
                time.sleep(sleep_for)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry failed without exception")

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt >= self._max_attempts:
            return False
        if isinstance(exc, RuntimeError):
            msg = str(exc).lower()
            if (
                "requires provider" in msg
                or "not configured" in msg
                or "unsupported llm provider" in msg
                or "requires llm" in msg
            ):
                return False
        return True
