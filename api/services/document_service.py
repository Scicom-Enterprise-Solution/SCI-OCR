import hashlib
import os
import uuid
from datetime import datetime, timezone

import cv2

from api.config import settings
from db import get_document, get_document_by_hash, insert_document
from document_inputs.loader import load_document_input
from path_utils import from_repo_relative, to_repo_relative


def ensure_storage_dirs() -> None:
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.previews_dir, exist_ok=True)
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.exports_dir, exist_ok=True)


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _write_preview_png(document_id: str, image_bgr) -> tuple[str, int, int]:
    preview_path = os.path.join(settings.previews_dir, f"{document_id}.png")
    cv2.imwrite(preview_path, image_bgr)
    h, w = image_bgr.shape[:2]
    return preview_path, w, h


def _document_artifacts_exist(record: dict) -> bool:
    stored_path = record.get("stored_path")
    preview_path = record.get("preview_path")
    if not stored_path or not preview_path:
        return False
    return (
        os.path.isfile(from_repo_relative(stored_path))
        and os.path.isfile(from_repo_relative(preview_path))
    )


def _restore_existing_document_artifacts(record: dict, file_bytes: bytes) -> dict:
    stored_path = from_repo_relative(record["stored_path"])
    preview_path = from_repo_relative(record["preview_path"])

    if not os.path.isfile(stored_path):
        _write_bytes(stored_path, file_bytes)

    if not os.path.isfile(preview_path):
        loaded = load_document_input(file_bytes=file_bytes, filename=record["filename"])
        restored_preview_path, preview_width, preview_height = _write_preview_png(record["id"], loaded.image_bgr)
        record["preview_path"] = to_repo_relative(restored_preview_path)
        record["preview_width"] = preview_width
        record["preview_height"] = preview_height

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
    preview_path, preview_width, preview_height = _write_preview_png(document_id, loaded.image_bgr)

    record = {
        "id": document_id,
        "filename": filename,
        "file_hash": file_hash,
        "source_type": loaded.source_type,
        "extension": loaded.extension,
        "stored_path": to_repo_relative(stored_path),
        "preview_path": to_repo_relative(preview_path),
        "preview_width": preview_width,
        "preview_height": preview_height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deduplicated": False,
    }
    return insert_document(settings.db_path, record)


def fetch_document(document_id: str) -> dict | None:
    return get_document(settings.db_path, document_id)


def fetch_document_preview_path(document_id: str) -> str:
    record = fetch_document(document_id)
    if record is None:
        raise FileNotFoundError(f"document not found: {document_id}")

    preview_path = record.get("preview_path")
    if not preview_path:
        raise FileNotFoundError(f"preview not available for document: {document_id}")

    absolute_preview_path = from_repo_relative(preview_path)
    if not os.path.isfile(absolute_preview_path):
        raise FileNotFoundError(f"preview file not found for document: {document_id}")

    return absolute_preview_path
