import json
import os

import cv2

from document_inputs import DocumentInputError, SUPPORTED_IMAGE_EXTS, load_document_input
from face_detection import orient_with_face_hint, extract_face_crop, draw_face_box
from mrz_rotation import detect_mrz_with_rotation_fallback
from report_utils import parse_mrz_td3, write_pipeline_report

import detect_mrz as stage2
import ocr_mrz as stage3
import preprocess_passport as stage1


def _make_logger(enabled: bool):
    return print if enabled else (lambda *args, **kwargs: None)


def build_reference_comparison(filename: str, line1: str, line2: str) -> dict:
    reference_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "samples",
        "reference_clean.json",
    )

    comparison = {
        "reference_available": False,
        "reference_file": os.path.abspath(reference_path),
        "filename": filename,
    }

    if not os.path.isfile(reference_path):
        return comparison

    try:
        with open(reference_path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except (OSError, json.JSONDecodeError):
        comparison["reference_error"] = "reference_file_unreadable"
        return comparison

    expected = refs.get(filename)
    if not isinstance(expected, dict):
        return comparison

    expected_line1 = expected.get("line1", "")
    expected_line2 = expected.get("line2", "")
    line1_match = line1 == expected_line1
    line2_match = line2 == expected_line2

    comparison.update({
        "reference_available": True,
        "line1_match": line1_match,
        "line2_match": line2_match,
        "exact_match": line1_match and line2_match,
        "expected": {
            "line1": expected_line1,
            "line2": expected_line2,
        },
    })
    return comparison


def print_reference_comparison(comparison: dict) -> None:
    if not comparison.get("reference_available"):
        print("[Reference] No reference entry for this sample.")
        return

    line1_match = comparison.get("line1_match", False)
    line2_match = comparison.get("line2_match", False)
    exact_match = comparison.get("exact_match", False)

    print(
        "[Reference] "
        f"{'PASS' if exact_match else 'FAIL'} "
        f"(line1={'PASS' if line1_match else 'FAIL'}, "
        f"line2={'PASS' if line2_match else 'FAIL'})"
    )

    if not line1_match:
        print(f"[Reference] Expected line1: {comparison['expected']['line1']}")
    if not line2_match:
        print(f"[Reference] Expected line2: {comparison['expected']['line2']}")


def build_output_prefix(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    if not stem:
        return "input"

    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)


def build_stage2_attempts(input_path: str, aligned_image, original_image):
    attempts = [("aligned_passport", aligned_image)]
    ext = os.path.splitext(input_path)[1].lower()

    if ext in SUPPORTED_IMAGE_EXTS and original_image is not None and original_image is not aligned_image:
        attempts.append(("original_input", original_image))

    return attempts


def process_document(
    input_path: str | None = None,
    *,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    use_face_hint: bool = True,
    emit_progress: bool = True,
) -> dict:
    log = _make_logger(emit_progress)
    loaded_input = load_document_input(
        input_path=input_path,
        file_bytes=file_bytes,
        filename=filename,
        dpi=stage1.DPI,
    )
    source_name = input_path or loaded_input.filename
    base_output_dir = os.getenv("OUTPUT_DIR", stage1.OUTPUT_DIR)
    output_prefix = build_output_prefix(loaded_input.filename)
    output_dir = os.path.join(base_output_dir, output_prefix)

    stage1.OUTPUT_DIR = output_dir
    stage2.OUTPUT_DIR = output_dir
    stage3.OUTPUT_DIR = output_dir

    os.makedirs(output_dir, exist_ok=True)

    report = {
        "status": "started",
        "input": {
            "path": os.path.abspath(input_path) if input_path else None,
            "filename": loaded_input.filename,
            "source_type": loaded_input.source_type,
            "ext": loaded_input.extension,
            "sample_name": output_prefix,
        },
        "output_dir": os.path.abspath(output_dir),
        "face": {
            "detected": False,
            "orientation_hint": None,
            "faces_count": 0,
            "bbox": None,
        },
        "mrz": {
            "detected": False,
            "detection_input": None,
            "detection_inputs_tried": [],
            "orientation_used": None,
            "bbox": None,
            "line_bboxes": [],
            "text": {"line1": "", "line2": ""},
            "parsed": {},
            "ocr": {},
        },
    }

    log("=" * 60)
    log("STAGE 1 — Input load/render + document alignment")
    log("=" * 60)

    if loaded_input.source_type == "pdf":
        log(f"[Stage 1] Input type: PDF ({source_name})")
        log(
            f"[Stage 1] Rendered first page at {stage1.DPI} DPI -> "
            f"{loaded_input.image_bgr.shape[1]}x{loaded_input.image_bgr.shape[0]} px (W x H)"
        )
    else:
        log(f"[Stage 1] Input type: image ({source_name})")
        log(
            f"[Stage 1] Loaded image at {loaded_input.image_bgr.shape[1]}x{loaded_input.image_bgr.shape[0]} px (W x H)"
        )

    img_bgr = loaded_input.image_bgr
    stage1.save(img_bgr, "rendered_page.png")

    edges, detection_scale = stage1.preprocess_image(img_bgr)
    edges_display = cv2.resize(
        edges,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    stage1.save(edges_display, "edges.png")

    quad_pts, contour_debug = stage1.find_document_contour(edges, img_bgr, detection_scale)
    stage1.save(contour_debug, "contours.png")

    if quad_pts is None:
        report["status"] = "failed"
        report["error"] = "stage1_document_contour_not_found"
        report["report_path"] = write_pipeline_report(output_dir, report)
        log(f"[Report] {report['report_path']}")
        log("\n[Pipeline] STAGE 1 FAILED: could not locate a document boundary.")
        log("           Check output/edges.png and output/contours.png for clues.")
        return report

    aligned = stage1.perspective_correction(img_bgr, quad_pts)
    stage1.save(aligned, "aligned_passport.png")

    face_result = {
        "image": aligned,
        "label": "original",
        "faces_count": 0,
        "face_bbox": None,
    }

    if use_face_hint:
        face_result = orient_with_face_hint(aligned)
        aligned_for_stage2 = face_result["image"]
    else:
        aligned_for_stage2 = aligned
        log("[Stage 1] Face hint disabled via USE_FACE_HINT=False")

    if face_result["label"] != "original":
        log(f"[Stage 1] Face-based orientation hint applied: {face_result['label']}")
        stage1.save(aligned_for_stage2, "aligned_passport.png")

    report["face"]["orientation_hint"] = face_result["label"]
    report["face"]["faces_count"] = int(face_result["faces_count"])

    if face_result["face_bbox"] is not None:
        face_crop = extract_face_crop(aligned_for_stage2, face_result["face_bbox"])
        if face_crop is not None and face_crop.size > 0:
            stage1.save(face_crop, "face_crop.png")

        face_annotated = draw_face_box(aligned_for_stage2, face_result["face_bbox"])
        stage1.save(face_annotated, "face_detected.png")

        report["face"]["detected"] = True
        report["face"]["bbox"] = [int(v) for v in face_result["face_bbox"]]
        log(f"[Stage 1] Face detected (count={face_result['faces_count']}).")
    else:
        if use_face_hint:
            log("[Stage 1] No face detected; continuing with MRZ-based orientation fallback.")
        else:
            log("[Stage 1] Face detection skipped; continuing with MRZ-based orientation fallback.")

    log("\n[Stage 1] Complete.\n")

    log("=" * 60)
    log("STAGE 2 — MRZ region detection")
    log("=" * 60)

    detection_result = None
    detection_input = None
    stage2_attempts = build_stage2_attempts(loaded_input.filename, aligned_for_stage2, img_bgr)

    for source_label, source_image in stage2_attempts:
        report["mrz"]["detection_inputs_tried"].append(source_label)
        log(f"[Stage 2] Detection source: {source_label}")
        detection_result = detect_mrz_with_rotation_fallback(source_image)
        if detection_result:
            detection_input = source_label
            break

        if len(stage2_attempts) > 1 and source_label != stage2_attempts[-1][0]:
            log(f"[Stage 2] MRZ not found on {source_label}; retrying next source.")

    if not detection_result:
        report["status"] = "failed"
        report["error"] = "stage2_mrz_not_detected"
        report["report_path"] = write_pipeline_report(output_dir, report)
        log(f"[Report] {report['report_path']}")
        log("\n[Pipeline] STAGE 2 FAILED: MRZ region not detected.")
        log("           Inspect the debug images in output/ for clues:")
        log("            mrz_blackhat.png  — should brighten the MRZ text area")
        log("            mrz_gradient.png  — should show a bright band at the bottom")
        log("            mrz_threshold.png — should show a thick bar at the bottom")
        log("            mrz_closed.png    — should show a solid rectangle near bottom")
        return report

    working_aligned, mrz_lines, mrz_bbox, used_orientation = detection_result

    if detection_input != "aligned_passport":
        log(f"[Stage 2] MRZ found using fallback source: {detection_input}")
        stage1.save(working_aligned, "aligned_passport.png")
    elif used_orientation != "original":
        log(f"[Stage 2] MRZ found after orientation correction: {used_orientation}")
        stage1.save(working_aligned, "aligned_passport.png")
    else:
        log("[Stage 2] MRZ found on original orientation")

    report["mrz"]["detected"] = True
    report["mrz"]["detection_input"] = detection_input
    report["mrz"]["orientation_used"] = used_orientation
    report["mrz"]["bbox"] = [int(v) for v in mrz_bbox]
    report["mrz"]["line_bboxes"] = [[int(v) for v in bbox] for bbox in mrz_lines]

    mrz_crop = stage2.crop_mrz(working_aligned, mrz_bbox)
    stage2.save(mrz_crop, "mrz_region.png")

    debug_img = stage2.draw_debug_boxes(working_aligned, mrz_lines, mrz_bbox)
    stage2.save(debug_img, "mrz_detected.png")

    log("\n[Stage 2] Complete.\n")

    log("=" * 60)
    log("STAGE 3 — MRZ OCR")
    log("=" * 60)

    mrz_clean = stage3.clean_mrz_image(mrz_crop)
    stage3.save(mrz_clean, "mrz_clean.png")

    ocr_result = stage3.run_ocr(mrz_crop)

    line1 = ""
    line2 = ""
    ocr_meta = {}

    if isinstance(ocr_result, tuple) and len(ocr_result) >= 2:
        line1, line2 = ocr_result[0], ocr_result[1]
        if len(ocr_result) >= 3 and isinstance(ocr_result[2], dict):
            ocr_meta = ocr_result[2]
    elif isinstance(ocr_result, str):
        line1 = ocr_result

    report["mrz"]["text"]["line1"] = line1
    report["mrz"]["text"]["line2"] = line2
    report["mrz"]["parsed"] = ocr_meta.get("parsed_fields") or parse_mrz_td3(line1, line2)
    report["mrz"]["ocr"] = ocr_meta
    report["reference_comparison"] = build_reference_comparison(loaded_input.filename, line1, line2)
    report["status"] = "success"

    report["report_path"] = write_pipeline_report(output_dir, report)
    log(f"[Report] {report['report_path']}")
    if emit_progress:
        print_reference_comparison(report["reference_comparison"])

    log("\n[Stage 3] Final MRZ:")
    log("-" * 60)
    if line1 or line2:
        log(line1)
        log(line2)
    else:
        log("(no text detected)")
    log("-" * 60)
    log("\n[Stage 3] Complete.\n")

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)

    out = os.path.abspath(stage1.OUTPUT_DIR)
    log(f"  aligned passport : {out}\\aligned_passport.png")
    log(f"  MRZ region       : {out}\\mrz_region.png")
    log(f"  MRZ annotated    : {out}\\mrz_detected.png")
    log(f"  MRZ cleaned      : {out}\\mrz_clean.png")
    log()
    log("Extracted MRZ text:")
    if line1 or line2:
        log(line1)
        log(line2)
    else:
        log("(no text)")
    log()
    if emit_progress:
        print_reference_comparison(report["reference_comparison"])

    return report
