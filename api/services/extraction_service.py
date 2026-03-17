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
from logger_utils import is_debug_enabled, log_event
from path_utils import from_repo_relative, to_repo_relative
from pipelines.mrz_pipeline import process_document

from .document_service import fetch_document


TRANSFORM_PAD_RATIO = 0.14


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


def _apply_transform(image_bgr, transform: dict | None):
    if not transform:
        return image_bgr

    angle = -float(transform.get("micro_rotation", 0.0) or 0.0)
    zoom = float(transform.get("zoom", 1.0) or 1.0)
    offset_x = float(transform.get("offset_x", 0.0) or 0.0)
    offset_y = float(transform.get("offset_y", 0.0) or 0.0)
    viewport_width = int(transform.get("viewport_width") or 0)
    viewport_height = int(transform.get("viewport_height") or 0)

    if (
        abs(angle) < 1e-6
        and abs(zoom - 1.0) < 1e-6
        and abs(offset_x) < 1e-6
        and abs(offset_y) < 1e-6
        and viewport_width <= 0
        and viewport_height <= 0
    ):
        return image_bgr

    h, w = image_bgr.shape[:2]
    pad_x = int(round(w * TRANSFORM_PAD_RATIO))
    pad_y = int(round(h * TRANSFORM_PAD_RATIO))
    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    padded_h, padded_w = padded.shape[:2]
    center = (padded_w / 2.0, padded_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, zoom)
    matrix[0, 2] += offset_x * w
    matrix[1, 2] += offset_y * h

    if viewport_width > 0 and viewport_height > 0:
        scale = min(viewport_width / float(w), viewport_height / float(h))
        render_center = (viewport_width / 2.0, viewport_height / 2.0)
        render_matrix = matrix.copy()
        render_matrix[0, 2] += render_center[0] - center[0] * scale
        render_matrix[1, 2] += render_center[1] - center[1] * scale
        render_matrix[0, 0] *= scale
        render_matrix[0, 1] *= scale
        render_matrix[1, 0] *= scale
        render_matrix[1, 1] *= scale
        return cv2.warpAffine(
            padded,
            render_matrix,
            (viewport_width, viewport_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    warped = cv2.warpAffine(
        padded,
        matrix,
        (padded_w, padded_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped[pad_y:pad_y + h, pad_x:pad_x + w]


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
        raise RuntimeError("Failed to encode corrected image as PNG.")
    return data.tobytes()


def run_backend_correction(
    file_bytes: bytes,
    *,
    filename: str,
    rotation: int,
    transform: dict | None,
    crop: dict | None,
) -> bytes:
    loaded = load_document_input(file_bytes=file_bytes, filename=filename)
    corrected = _apply_rotation(loaded.image_bgr, rotation)
    corrected = _apply_transform(corrected, transform)
    corrected = _apply_crop(corrected, crop)
    return _encode_png(corrected)


def should_run_correction(input_mode: str, enable_correction: bool | None) -> bool:
    normalized = (input_mode or "raw").strip().lower() or "raw"
    if normalized not in {"raw", "frontend"}:
        raise ValueError("input_mode must be 'raw' or 'frontend'")
    if enable_correction:
        return True
    if normalized == "raw":
        return True
    return False


def _validate_frontend_input(file_bytes: bytes, filename: str, input_mode: str) -> None:
    if input_mode != "frontend":
        return
    loaded = load_document_input(file_bytes=file_bytes, filename=filename)
    height, width = loaded.image_bgr.shape[:2]
    if width < 600 or height < 400:
        raise ValueError("frontend input image is too small; minimum size is 600x400")


def _enforce_frontend_geometry_boundary(
    input_mode: str,
    *,
    crop: dict | None,
    transform: dict | None,
    rotation: int,
) -> tuple[dict | None, dict | None, int]:
    if input_mode != "frontend":
        return crop, transform, rotation
    if crop is not None or transform is not None or rotation != 0:
        raise ValueError(
            "frontend mode does not accept crop/transform/rotation; send final prepared image instead"
        )
    return None, None, 0


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
    crop, transform, rotation = _enforce_frontend_geometry_boundary(
        input_mode_normalized,
        crop=crop,
        transform=transform,
        rotation=rotation,
    )
    correction_applied = should_run_correction(input_mode_normalized, enable_correction)
    variant_profile = "clean" if input_mode_normalized == "frontend" else "aggressive"

    pipeline_bytes = _read_bytes(stored_path)
    _validate_frontend_input(pipeline_bytes, document["filename"], input_mode_normalized)
    if correction_applied:
        pipeline_bytes = run_backend_correction(
            pipeline_bytes,
            filename=document["filename"],
            rotation=rotation,
            transform=transform,
            crop=crop,
        )

    log_event(
        "api_extraction_correction",
        input_mode=input_mode_normalized,
        enable_correction=enable_correction,
        correction_applied=correction_applied,
        variant_profile=variant_profile,
    )

    extraction_id = uuid.uuid4().hex
    pipeline_filename = f"{extraction_id}_{os.path.splitext(document['filename'])[0]}.png"
    if is_debug_enabled():
        report = process_document(
            file_bytes=pipeline_bytes,
            filename=pipeline_filename,
            use_face_hint=use_face_hint if correction_applied else False,
            skip_document_alignment=not correction_applied,
            strict_input_orientation=not correction_applied,
            prealigned_input=not correction_applied,
            emit_progress=True,
        )
    else:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            report = process_document(
                file_bytes=pipeline_bytes,
                filename=pipeline_filename,
                use_face_hint=use_face_hint if correction_applied else False,
                skip_document_alignment=not correction_applied,
                strict_input_orientation=not correction_applied,
                prealigned_input=not correction_applied,
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
        "enable_correction": enable_correction,
        "correction_applied": correction_applied,
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
