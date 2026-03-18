from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas import LLMChatRequest, LLMChatResponse
from api.services.llm_service import is_llm_enabled, run_chat_completion
from logger_utils import log_event


router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/health")
def llm_health() -> dict[str, bool | str]:
    return {
        "enabled": is_llm_enabled(),
        "base_url": bool(settings.llm_api_base_url),
        "model": bool(settings.llm_model),
    }


@router.post("/chat", response_model=LLMChatResponse)
def chat(request: LLMChatRequest) -> LLMChatResponse:
    log_event(
        "api_llm_chat_started",
        message_count=len(request.messages),
        requested_model=request.model,
    )
    try:
        result = run_chat_completion(
            messages=[message.model_dump() for message in request.messages],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        log_event("api_llm_chat_failed", level="error", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        log_event("api_llm_chat_failed", level="error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log_event(
        "api_llm_chat_finished",
        model=result["model"],
        content_length=len(result["content"]),
    )
    return LLMChatResponse(**result)
