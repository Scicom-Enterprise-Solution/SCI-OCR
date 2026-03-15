import os
import shutil
import uuid
from datetime import datetime, timezone
import io
from contextlib import redirect_stderr, redirect_stdout

import cv2

from api.config import settings
from db import get_extraction, insert_extraction
from document_inputs.loader import load_document_input
from logger_utils import is_debug_enabled
from path_utils import from_repo_relative, to_repo_relative
from pipelines.mrz_pipeline import process_document

from .document_service import fetch_document


def _read_bytes(path: str) -> bytes:
    resolved = from_repo_relative(path)
    with open(resolved, "rb") as f:
        return f.read()


def _apply_rotation(image_bgr, rotation: int):
    if rotation == 0:
        return image_bgr
    if rotation == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be one of 0, 90, 180, 270")


def _apply_crop(image_bgr, crop: dict | None):
    if not crop:
        return image_bgr

    h, w = image_bgr.shape[:2]
    x1 = int(round(crop["x"] * w))
    y1 = int(round(crop["y"] * h))
    x2 = int(round((crop["x"] + crop["width"]) * w))
    y2 = int(round((crop["y"] + crop["height"]) * h))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return image_bgr[y1:y2, x1:x2]


def _encode_png(image_bgr) -> bytes:
    ok, data = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode cropped image as PNG.")
    return data.tobytes()


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
    crop: dict | None,
    rotation: int,
    use_face_hint: bool,
) -> dict:
    document = fetch_document(document_id)
    if document is None:
        raise FileNotFoundError(f"Unknown document_id: {document_id}")

    original_bytes = _read_bytes(document["stored_path"])
    loaded = load_document_input(file_bytes=original_bytes, filename=document["filename"])
    working = _apply_rotation(loaded.image_bgr, rotation)
    working = _apply_crop(working, crop)

    extraction_id = uuid.uuid4().hex
    pipeline_filename = f"{extraction_id}_{os.path.splitext(document['filename'])[0]}.png"
    pipeline_bytes = _encode_png(working)
    if is_debug_enabled():
        report = process_document(
            file_bytes=pipeline_bytes,
            filename=pipeline_filename,
            use_face_hint=use_face_hint,
            emit_progress=True,
        )
    else:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            report = process_document(
                file_bytes=pipeline_bytes,
                filename=pipeline_filename,
                use_face_hint=use_face_hint,
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
        "crop": crop,
        "rotation": rotation,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return insert_extraction(settings.db_path, record)


def fetch_extraction(extraction_id: str) -> dict | None:
    return get_extraction(settings.db_path, extraction_id)
