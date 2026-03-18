import json
from typing import Any
from urllib import error, request

from api.config import settings


def is_llm_enabled() -> bool:
    return bool(settings.llm_api_base_url and settings.llm_model)


def _build_chat_url() -> str:
    base_url = settings.llm_api_base_url.rstrip("/")
    if not base_url:
        raise ValueError("LLM_API_BASE_URL is not configured")
    return f"{base_url}/chat/completions"


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include any choices")

    message = choices[0].get("message") or {}
    content = _coerce_message_text(message.get("content"))
    if content:
        return content

    raise RuntimeError("LLM response did not include message content")


def run_chat_completion(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if not is_llm_enabled():
        raise ValueError("LLM integration is not configured")

    payload: dict[str, Any] = {
        "model": (model or settings.llm_model).strip(),
        "messages": messages,
    }
    if not payload["model"]:
        raise ValueError("No LLM model was provided")
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    req = request.Request(
        _build_chat_url(),
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.llm_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body or str(exc)
        raise RuntimeError(f"LLM request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc

    try:
        response_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response was not valid JSON") from exc

    return {
        "provider": settings.llm_api_base_url,
        "model": response_payload.get("model") or payload["model"],
        "content": _extract_content(response_payload),
        "raw": response_payload,
    }
