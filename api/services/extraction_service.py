import os
import shutil
import uuid
from datetime import datetime, timezone
import io
from contextlib import redirect_stderr, redirect_stdout

from api.config import settings
from db import get_extraction, insert_extraction
from logger_utils import is_debug_enabled
from path_utils import from_repo_relative, to_repo_relative
from pipelines.mrz_pipeline import process_document

from .document_service import fetch_document


def _read_bytes(path: str) -> bytes:
    resolved = from_repo_relative(path)
    with open(resolved, "rb") as f:
        return f.read()


def _relocate_api_report(report_path: str | None, document_id: str) -> str | None:
    if not report_path:
        return None

    source = os.path.abspath(report_path)
    if not os.path.isfile(source):
        return to_repo_relative(source)

    os.makedirs(settings.reports_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%d%m%y%H%M%S")
    target = os.path.join(settings.reports_dir, f"{document_id}_{timestamp}.json")
    suffix = 1
    while os.path.exists(target):
        target = os.path.join(settings.reports_dir, f"{document_id}_{timestamp}_{suffix}.json")
        suffix += 1
    shutil.move(source, target)
    return to_repo_relative(target)


def create_extraction(
    *,
    document_id: str,
    input_mode: str,
    enable_correction: bool | None,
    crop: dict | None,
    rotation: int,
    transform: dict | None,
    use_face_hint: bool,
) -> dict:
    document = fetch_document(document_id)
    if document is None:
        raise FileNotFoundError(f"Unknown document_id: {document_id}")

    stored_path = document.get("stored_path")
    if not stored_path:
        raise FileNotFoundError(f"Stored file is not available for document_id: {document_id}")

    input_mode_normalized = (input_mode or "raw").strip().lower() or "raw"
    if input_mode_normalized not in {"raw", "frontend"}:
        raise ValueError("input_mode must be 'raw' or 'frontend'")

    correction_enabled = (
        bool(enable_correction)
        if enable_correction is not None
        else input_mode_normalized != "frontend"
    )

    pipeline_bytes = _read_bytes(stored_path)

    extraction_id = uuid.uuid4().hex
    pipeline_filename = f"{extraction_id}_{os.path.splitext(document['filename'])[0]}.png"
    if is_debug_enabled():
        report = process_document(
            file_bytes=pipeline_bytes,
            filename=pipeline_filename,
            use_face_hint=use_face_hint if correction_enabled else False,
            skip_document_alignment=not correction_enabled,
            strict_input_orientation=not correction_enabled,
            prealigned_input=not correction_enabled,
            emit_progress=True,
        )
    else:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            report = process_document(
                file_bytes=pipeline_bytes,
                filename=pipeline_filename,
                use_face_hint=use_face_hint if correction_enabled else False,
                skip_document_alignment=not correction_enabled,
                strict_input_orientation=not correction_enabled,
                prealigned_input=not correction_enabled,
                emit_progress=False,
            )
    report_path = _relocate_api_report(report.get("report_path"), document_id)

    record = {
        "id": extraction_id,
        "document_id": document_id,
        "status": report.get("status", "unknown"),
        "filename": document["filename"],
        "line1": report.get("mrz", {}).get("text", {}).get("line1", ""),
        "line2": report.get("mrz", {}).get("text", {}).get("line2", ""),
        "parsed": report.get("mrz", {}).get("parsed", {}),
        "report_path": report_path,
        "duration_ms": report.get("duration_ms"),
        "crop": None if input_mode_normalized == "frontend" else crop,
        "rotation": rotation,
        "transform": None if input_mode_normalized == "frontend" else transform,
        "input_mode": input_mode_normalized,
        "enable_correction": correction_enabled,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return insert_extraction(settings.db_path, record)


def fetch_extraction(extraction_id: str) -> dict | None:
    return get_extraction(settings.db_path, extraction_id)


def fetch_extraction_report_path(extraction_id: str) -> str:
    record = fetch_extraction(extraction_id)
    if record is None:
        raise FileNotFoundError(f"extraction not found: {extraction_id}")

    report_path = record.get("report_path")
    if not report_path:
        raise FileNotFoundError(f"report not available for extraction: {extraction_id}")

    absolute_report_path = from_repo_relative(report_path)
    if not os.path.isfile(absolute_report_path):
        raise FileNotFoundError(f"report file not found for extraction: {extraction_id}")

    return absolute_report_path
