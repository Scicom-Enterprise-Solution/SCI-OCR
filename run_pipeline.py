"""
Passport MRZ OCR — Full Pipeline
PDF → aligned document image → detected & cropped MRZ strip

Runs both stages in sequence:
    Stage 1 (preprocess_passport): input load/render → contour detection → perspective warp
    Stage 2 (detect_mrz):          blackhat → gradient → threshold → MRZ crop
    Stage 3 (ocr_mrz):             MRZ cleaning → OCR → parsing/reporting

Usage:
    python run_pipeline.py <path/to/passport.pdf>
    python run_pipeline.py <path/to/passport.png>

Output (all saved to output/):
    <input_name>/rendered_page.png     — raw render / loaded image
    <input_name>/edges.png             — Canny edges (Stage 1 debug)
    <input_name>/contours.png          — detected document contour (Stage 1 debug)
    <input_name>/aligned_passport.png  — perspective-corrected passport page
    <input_name>/mrz_blackhat.png      — blackhat morphology (Stage 2 debug)
    <input_name>/mrz_gradient.png      — horizontal gradient (Stage 2 debug)
    <input_name>/mrz_threshold.png     — Otsu threshold (Stage 2 debug)
    <input_name>/mrz_closed.png        — morphological closing (Stage 2 debug)
    <input_name>/mrz_detected.png      — annotated passport with MRZ bounding boxes
    <input_name>/mrz_region.png        — final cropped MRZ strip (pipeline output)
"""

import os
import sys
import cv2

from env_utils import load_env_file

load_env_file()

USE_FACE_HINT = os.getenv("USE_FACE_HINT", "True").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Import both stages. Each module is self-contained; we call their functions
# individually rather than running them as scripts, so we can pass data
# directly between stages without touching disk mid-pipeline.
# ---------------------------------------------------------------------------

import preprocess_passport as stage1
import detect_mrz as stage2
import ocr_mrz as stage3

from face_detection import orient_with_face_hint, extract_face_crop, draw_face_box
from mrz_rotation import detect_mrz_with_rotation_fallback
from report_utils import write_pipeline_report, parse_mrz_td3


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_stage1_input(input_path: str):
    """Load either a PDF page or an image file into a BGR numpy image."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        print(f"[Stage 1] Input type: PDF ({input_path})")
        return stage1.render_pdf_page(input_path, dpi=stage1.DPI)

    if ext in SUPPORTED_IMAGE_EXTS:
        print(f"[Stage 1] Input type: image ({input_path})")
        img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[ERROR] Unable to read image: {input_path}")
            sys.exit(1)
        h, w = img_bgr.shape[:2]
        print(f"[Stage 1] Loaded image at {w}x{h} px (W×H)")
        return img_bgr

    print(f"[ERROR] Unsupported input type: '{ext}'")
    print("        Supported: .pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp")
    sys.exit(1)


def build_output_prefix(input_path: str) -> str:
    """Use input filename stem as a stable prefix for all output artifacts."""
    stem = os.path.splitext(os.path.basename(input_path))[0].strip()
    if not stem:
        return "input"

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return safe


def build_stage2_attempts(input_path: str, aligned_image, original_image):
    """Choose Stage 2 MRZ detection sources in priority order."""
    attempts = [("aligned_passport", aligned_image)]
    ext = os.path.splitext(input_path)[1].lower()

    if ext in SUPPORTED_IMAGE_EXTS and original_image is not None and original_image is not aligned_image:
        attempts.append(("original_input", original_image))

    return attempts


def main() -> None:
    env_input_path = os.getenv("PDF_PATH", stage1.PDF_PATH)
    input_path = sys.argv[1] if len(sys.argv) > 1 else env_input_path

    base_output_dir = os.getenv("OUTPUT_DIR", stage1.OUTPUT_DIR)
    output_prefix = build_output_prefix(input_path)
    output_dir = os.path.join(base_output_dir, output_prefix)

    stage1.OUTPUT_DIR = output_dir
    stage2.OUTPUT_DIR = output_dir
    stage3.OUTPUT_DIR = output_dir

    os.makedirs(output_dir, exist_ok=True)

    report = {
        "status": "started",
        "input": {
            "path": os.path.abspath(input_path),
            "ext": os.path.splitext(input_path)[1].lower(),
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
            "text": {
                "line1": "",
                "line2": "",
            },
            "parsed": {},
            "ocr": {},
        },
    }

    # ======================================================================
    # STAGE 1 — Input file → aligned passport image
    # ======================================================================
    print("=" * 60)
    print("STAGE 1 — Input load/render + document alignment")
    print("=" * 60)

    # Step 1 — Load/render input
    img_bgr = load_stage1_input(input_path)
    stage1.save(img_bgr, "rendered_page.png")

    # Step 2 — Edge detection (downscaled)
    edges, detection_scale = stage1.preprocess_image(img_bgr)
    edges_display = cv2.resize(
        edges,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    stage1.save(edges_display, "edges.png")

    # Step 3 — Contour detection → quadrilateral
    quad_pts, contour_debug = stage1.find_document_contour(
        edges, img_bgr, detection_scale
    )
    stage1.save(contour_debug, "contours.png")

    if quad_pts is None:
        report["status"] = "failed"
        report["error"] = "stage1_document_contour_not_found"
        report_path = write_pipeline_report(output_dir, report)
        print(f"[Report] {report_path}")
        print("\n[Pipeline] STAGE 1 FAILED: could not locate a document boundary.")
        print("           Check output/edges.png and output/contours.png for clues.")
        sys.exit(1)

    # Step 4 — Perspective correction
    aligned = stage1.perspective_correction(img_bgr, quad_pts)
    stage1.save(aligned, "aligned_passport.png")

    # Face-based orientation hint + face extraction for downstream checks.
    # Always initialise a default result so the rest of the pipeline works
    # even when face hinting is disabled.
    face_result = {
        "image": aligned,
        "label": "original",
        "faces_count": 0,
        "face_bbox": None,
    }

    if USE_FACE_HINT:
        face_result = orient_with_face_hint(aligned)
        aligned_for_stage2 = face_result["image"]
    else:
        aligned_for_stage2 = aligned
        print("[Stage 1] Face hint disabled via USE_FACE_HINT=False")

    if face_result["label"] != "original":
        print(f"[Stage 1] Face-based orientation hint applied: {face_result['label']}")
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
        print(f"[Stage 1] Face detected (count={face_result['faces_count']}).")
    else:
        if USE_FACE_HINT:
            print("[Stage 1] No face detected; continuing with MRZ-based orientation fallback.")
        else:
            print("[Stage 1] Face detection skipped; continuing with MRZ-based orientation fallback.")

    print("\n[Stage 1] Complete.\n")

    # ======================================================================
    # STAGE 2 — Aligned image → MRZ strip
    # ======================================================================
    print("=" * 60)
    print("STAGE 2 — MRZ region detection")
    print("=" * 60)

    detection_result = None
    detection_input = None
    stage2_attempts = build_stage2_attempts(input_path, aligned_for_stage2, img_bgr)

    for source_label, source_image in stage2_attempts:
        report["mrz"]["detection_inputs_tried"].append(source_label)
        print(f"[Stage 2] Detection source: {source_label}")
        detection_result = detect_mrz_with_rotation_fallback(source_image)
        if detection_result:
            detection_input = source_label
            break

        if len(stage2_attempts) > 1 and source_label != stage2_attempts[-1][0]:
            print(f"[Stage 2] MRZ not found on {source_label}; retrying next source.")

    if not detection_result:
        report["status"] = "failed"
        report["error"] = "stage2_mrz_not_detected"
        report_path = write_pipeline_report(output_dir, report)
        print(f"[Report] {report_path}")
        print("\n[Pipeline] STAGE 2 FAILED: MRZ region not detected.")
        print("           Inspect the debug images in output/ for clues:")
        print("            mrz_blackhat.png  — should brighten the MRZ text area")
        print("            mrz_gradient.png  — should show a bright band at the bottom")
        print("            mrz_threshold.png — should show a thick bar at the bottom")
        print("            mrz_closed.png    — should show a solid rectangle near bottom")
        sys.exit(1)

    working_aligned, mrz_lines, mrz_bbox, used_orientation = detection_result

    if detection_input != "aligned_passport":
        print(f"[Stage 2] MRZ found using fallback source: {detection_input}")
        stage1.save(working_aligned, "aligned_passport.png")
    elif used_orientation != "original":
        print(f"[Stage 2] MRZ found after orientation correction: {used_orientation}")
        stage1.save(working_aligned, "aligned_passport.png")
    else:
        print("[Stage 2] MRZ found on original orientation")

    report["mrz"]["detected"] = True
    report["mrz"]["detection_input"] = detection_input
    report["mrz"]["orientation_used"] = used_orientation
    report["mrz"]["bbox"] = [int(v) for v in mrz_bbox]
    report["mrz"]["line_bboxes"] = [[int(v) for v in bbox] for bbox in mrz_lines]

    # Step 7 — Merge the line bboxes and crop
    mrz_crop = stage2.crop_mrz(working_aligned, mrz_bbox)
    stage2.save(mrz_crop, "mrz_region.png")

    # Step 8 — Debug visualisation
    debug_img = stage2.draw_debug_boxes(working_aligned, mrz_lines, mrz_bbox)
    stage2.save(debug_img, "mrz_detected.png")

    print("\n[Stage 2] Complete.\n")

    # ======================================================================
    # STAGE 3 — MRZ strip → cleaned image → OCR text
    # ======================================================================
    print("=" * 60)
    print("STAGE 3 — MRZ OCR")
    print("=" * 60)

    # Step 1 — Clean preview/debug image
    mrz_clean = stage3.clean_mrz_image(mrz_crop)
    stage3.save(mrz_clean, "mrz_clean.png")

    # Step 2 — OCR + parsing
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
    report["status"] = "success"

    report_path = write_pipeline_report(output_dir, report)
    print(f"[Report] {report_path}")

    print("\n[Stage 3] Final MRZ:")
    print("-" * 60)
    if line1 or line2:
        print(line1)
        print(line2)
    else:
        print("(no text detected)")
    print("-" * 60)
    print("\n[Stage 3] Complete.\n")

    # ======================================================================
    # Summary
    # ======================================================================
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    out = os.path.abspath(stage1.OUTPUT_DIR)
    print(f"  aligned passport : {out}\\aligned_passport.png")
    print(f"  MRZ region       : {out}\\mrz_region.png")
    print(f"  MRZ annotated    : {out}\\mrz_detected.png")
    print(f"  MRZ cleaned      : {out}\\mrz_clean.png")
    print()
    print("Extracted MRZ text:")
    if line1 or line2:
        print(line1)
        print(line2)
    else:
        print("(no text)")


if __name__ == "__main__":
    main()