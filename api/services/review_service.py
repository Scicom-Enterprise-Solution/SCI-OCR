from datetime import datetime, timezone

from api.config import settings
from db import list_references, upsert_reference

from .document_service import fetch_document


def save_reference_correction(
    *,
    document_id: str,
    line1: str,
    line2: str,
    notes: str = "",
) -> dict:
    document = fetch_document(document_id)
    if document is None:
        raise FileNotFoundError(f"Unknown document_id: {document_id}")

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "document_id": document_id,
        "filename": document["filename"],
        "line1": line1,
        "line2": line2,
        "notes": notes,
        "created_at": now,
        "updated_at": now,
    }
    return upsert_reference(settings.db_path, record)


def fetch_reference_corrections() -> list[dict]:
    return list_references(settings.db_path)
