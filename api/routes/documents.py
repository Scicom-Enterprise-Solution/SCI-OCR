from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.services.document_service import fetch_document_preview_path
from logger_utils import log_event


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/{document_id}/preview")
def get_document_preview(document_id: str) -> FileResponse:
    try:
        preview_path = fetch_document_preview_path(document_id)
    except FileNotFoundError as exc:
        log_event(
            "api_document_preview_failed",
            level="error",
            document_id=document_id,
            error=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_event(
        "api_document_preview_served",
        document_id=document_id,
        preview_path=preview_path,
    )
    return FileResponse(preview_path, media_type="image/png")
