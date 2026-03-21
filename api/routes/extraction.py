from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas import ExtractionRequest, ExtractionResponse
from api.services.extraction_service import create_extraction, fetch_extraction_report_path
from logger_utils import log_event


router = APIRouter(prefix="/api/extractions", tags=["extractions"])


@router.post("", response_model=ExtractionResponse)
def extract_document(request: ExtractionRequest) -> ExtractionResponse:
    log_event(
        "api_extraction_started",
        document_id=request.document_id,
        input_mode=request.input_mode,
        enable_correction=request.enable_correction,
        has_crop=request.crop is not None,
        rotation=request.rotation,
        use_face_hint=request.use_face_hint,
    )
    try:
        record = create_extraction(
            document_id=request.document_id,
            input_mode=request.input_mode,
            enable_correction=request.enable_correction,
            crop=request.crop.model_dump() if request.crop else None,
            rotation=request.rotation,
            transform=request.transform.model_dump() if request.transform else None,
            use_face_hint=request.use_face_hint,
        )
    except FileNotFoundError as exc:
        log_event(
            "api_extraction_failed",
            level="error",
            document_id=request.document_id,
            error=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log_event(
            "api_extraction_failed",
            level="error",
            document_id=request.document_id,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event(
        "api_extraction_finished",
        extraction_id=record["id"],
        document_id=record["document_id"],
        filename=record["filename"],
        status=record["status"],
        duration_ms=record.get("duration_ms"),
        report_path=record.get("report_path"),
    )

    return ExtractionResponse(
        extraction_id=record["id"],
        status=record["status"],
        filename=record["filename"],
        line1=record["line1"],
        line2=record["line2"],
        parsed=record.get("parsed") or {},
        confidence=record.get("confidence"),
        duration_ms=record.get("duration_ms"),
        report_path=record.get("report_path"),
        document_id=record["document_id"],
    )


@router.get("/{extraction_id}/report")
def get_extraction_report(extraction_id: str) -> FileResponse:
    try:
        report_path = fetch_extraction_report_path(extraction_id)
    except FileNotFoundError as exc:
        log_event(
            "api_extraction_report_failed",
            level="error",
            extraction_id=extraction_id,
            error=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_event(
        "api_extraction_report_served",
        extraction_id=extraction_id,
        report_path=report_path,
    )
    return FileResponse(report_path, media_type="application/json")
