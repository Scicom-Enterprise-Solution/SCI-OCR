from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import UploadResponse
from api.services.document_service import create_document
from logger_utils import log_event


router = APIRouter(prefix="/api/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    log_event(
        "api_upload_started",
        filename=filename,
        bytes_received=len(payload),
    )

    try:
        record = create_document(payload, filename)
    except Exception as exc:
        log_event(
            "api_upload_failed",
            level="error",
            filename=filename,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event(
        "api_upload_saved",
        document_id=record["id"],
        filename=record["filename"],
        file_hash=record.get("file_hash"),
        deduplicated=bool(record.get("deduplicated")),
        source_type=record["source_type"],
        preview_path=record["preview_path"],
        preview_width=record["preview_width"],
        preview_height=record["preview_height"],
    )

    return UploadResponse(
        document_id=record["id"],
        filename=record["filename"],
        source_type=record["source_type"],
        extension=record["extension"],
        file_hash=record.get("file_hash"),
        deduplicated=bool(record.get("deduplicated")),
        preview_path=record["preview_path"],
        preview_width=record["preview_width"],
        preview_height=record["preview_height"],
    )
