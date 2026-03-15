from fastapi import APIRouter, HTTPException

from api.schemas import ReferenceCorrectionRequest, ReferenceCorrectionResponse
from api.services.review_service import fetch_reference_corrections, save_reference_correction


router = APIRouter(prefix="/api/references", tags=["references"])


@router.get("", response_model=list[ReferenceCorrectionResponse])
def list_reference_corrections() -> list[ReferenceCorrectionResponse]:
    rows = fetch_reference_corrections()
    return [ReferenceCorrectionResponse(**row) for row in rows]


@router.post("", response_model=ReferenceCorrectionResponse)
def save_reference(request: ReferenceCorrectionRequest) -> ReferenceCorrectionResponse:
    try:
        row = save_reference_correction(
            document_id=request.document_id,
            line1=request.line1,
            line2=request.line2,
            notes=request.notes,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ReferenceCorrectionResponse(**row)
