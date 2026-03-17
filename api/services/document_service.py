import hashlib
import os
import uuid
from datetime import datetime, timezone

from api.config import settings
from db import get_document, get_document_by_hash, insert_document
from document_inputs.loader import load_document_input
from path_utils import from_repo_relative, to_repo_relative


def ensure_storage_dirs() -> None:
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.exports_dir, exist_ok=True)


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _document_artifacts_exist(record: dict) -> bool:
    stored_path = record.get("stored_path")
    if not stored_path:
        return False
    return os.path.isfile(from_repo_relative(stored_path))


def _restore_existing_document_artifacts(record: dict, file_bytes: bytes) -> dict:
    stored_path = from_repo_relative(record["stored_path"])

    if not os.path.isfile(stored_path):
        _write_bytes(stored_path, file_bytes)

    return record


def create_document(file_bytes: bytes, filename: str) -> dict:
    ensure_storage_dirs()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = get_document_by_hash(settings.db_path, file_hash)
    if existing is not None:
        if not _document_artifacts_exist(existing):
            existing = _restore_existing_document_artifacts(existing, file_bytes)
        existing["deduplicated"] = True
        return existing

    document_id = uuid.uuid4().hex
    extension = os.path.splitext(filename)[1].lower()
    stored_filename = f"{document_id}{extension}"
    stored_path = os.path.join(settings.uploads_dir, stored_filename)
    _write_bytes(stored_path, file_bytes)

    loaded = load_document_input(file_bytes=file_bytes, filename=filename)
    preview_height, preview_width = loaded.image_bgr.shape[:2]

    record = {
        "id": document_id,
        "filename": filename,
        "file_hash": file_hash,
        "source_type": loaded.source_type,
        "extension": loaded.extension,
        "stored_path": to_repo_relative(stored_path),
        "preview_path": to_repo_relative(stored_path),
        "preview_width": preview_width,
        "preview_height": preview_height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deduplicated": False,
    }
    return insert_document(settings.db_path, record)


def fetch_document(document_id: str) -> dict | None:
    return get_document(settings.db_path, document_id)
