"""
Passport MRZ OCR — Full Pipeline
PDF → aligned document image → detected & cropped MRZ strip

Runs both stages in sequence:
  Stage 1 (preprocess_passport): PDF render → contour detection → perspective warp
  Stage 2 (detect_mrz):          blackhat → gradient → threshold → MRZ crop

Usage:
    python run_pipeline.py <path/to/passport.pdf>

Output (all saved to output/):
    rendered_page.png     — raw PDF render at 300 DPI
    edges.png             — Canny edges (Stage 1 debug)
    contours.png          — detected document contour (Stage 1 debug)
    aligned_passport.png  — perspective-corrected passport page
    mrz_blackhat.png      — blackhat morphology (Stage 2 debug)
    mrz_gradient.png      — horizontal gradient (Stage 2 debug)
    mrz_threshold.png     — Otsu threshold (Stage 2 debug)
    mrz_closed.png        — morphological closing (Stage 2 debug)
    mrz_detected.png      — annotated passport with MRZ bounding boxes
    mrz_region.png        — final cropped MRZ strip (pipeline output)
"""

import os
import sys
from env_utils import load_env_file


load_env_file()

# ---------------------------------------------------------------------------
# Import both stages.  Each module is self-contained; we call their functions
# individually rather than running them as scripts, so we can pass data
# directly between stages without touching disk mid-pipeline.
# ---------------------------------------------------------------------------

import preprocess_passport as stage1
import detect_mrz          as stage2
import ocr_mrz             as stage3


def main() -> None:
    env_pdf_path = os.getenv("PDF_PATH", stage1.PDF_PATH)
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else env_pdf_path

    output_dir = os.getenv("OUTPUT_DIR", stage1.OUTPUT_DIR)
    stage1.OUTPUT_DIR = output_dir
    stage2.OUTPUT_DIR = output_dir
    stage3.OUTPUT_DIR = output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # STAGE 1 — PDF → aligned passport image
    # ======================================================================
    print("=" * 60)
    print("STAGE 1 — PDF render + document alignment")
    print("=" * 60)

    # Step 1 — Render PDF
    img_bgr = stage1.render_pdf_page(pdf_path, dpi=stage1.DPI)
    stage1.save(img_bgr, "rendered_page.png")

    # Step 2 — Edge detection (downscaled)
    edges, detection_scale = stage1.preprocess_image(img_bgr)
    edges_display = __import__("cv2").resize(
        edges,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=__import__("cv2").INTER_NEAREST,
    )
    stage1.save(edges_display, "edges.png")

    # Step 3 — Contour detection → quadrilateral
    quad_pts, contour_debug = stage1.find_document_contour(
        edges, img_bgr, detection_scale
    )
    stage1.save(contour_debug, "contours.png")

    if quad_pts is None:
        print("\n[Pipeline] STAGE 1 FAILED: could not locate a document boundary.")
        print("           Check output/edges.png and output/contours.png for clues.")
        sys.exit(1)

    # Step 4 — Perspective correction
    aligned = stage1.perspective_correction(img_bgr, quad_pts)
    stage1.save(aligned, "aligned_passport.png")

    print("\n[Stage 1] Complete.\n")

    # ======================================================================
    # STAGE 2 — Aligned image → MRZ strip
    # ======================================================================
    print("=" * 60)
    print("STAGE 2 — MRZ region detection")
    print("=" * 60)

    img_h, img_w = aligned.shape[:2]

    # Step 1 — Grayscale (aligned is already BGR from Stage 1)
    import cv2, numpy as np
    img_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    print(f"[Step 1] Passport image: {img_w}×{img_h} px (W×H)")

    # Restrict processing to the bottom region where the MRZ always sits.
    # Skipping the top avoids spurious wide contours from portrait / text fields.
    roi_top = int(img_h * stage2.ROI_TOP_FRACTION)
    roi     = img_gray[roi_top:, :]
    print(f"[Step 1] ROI: rows {roi_top}–{img_h}  "
          f"(bottom {100*(1-stage2.ROI_TOP_FRACTION):.0f}% of image)")

    # Step 2 — Blackhat morphology (on ROI)
    blackhat = stage2.apply_blackhat(roi)
    stage2.save(blackhat, "mrz_blackhat.png")

    # Step 3 — Horizontal gradient
    gradient = stage2.compute_gradient(blackhat)
    stage2.save(gradient, "mrz_gradient.png")

    # Step 4 — Threshold
    thresh = stage2.threshold_image(gradient)
    stage2.save(thresh, "mrz_threshold.png")

    # Step 5 — Morphological closing + dilation
    closed = stage2.close_and_dilate(thresh)
    stage2.save(closed, "mrz_closed.png")

    # Step 6 — Contour detection and MRZ line filtering
    mrz_lines, _ = stage2.find_mrz_contours(closed, img_h, img_w,
                                             roi_y_offset=roi_top)

    if not mrz_lines:
        print("\n[Pipeline] STAGE 2 FAILED: MRZ region not detected.")
        print("           Inspect the debug images in output/ for clues:")
        print("            mrz_blackhat.png  — should brighten the MRZ text area")
        print("            mrz_gradient.png  — should show a bright band at the bottom")
        print("            mrz_threshold.png — should show a thick bar at the bottom")
        print("            mrz_closed.png    — should show a solid rectangle near bottom")
        sys.exit(1)

    # Step 7 — Merge the two line bboxes and crop
    mrz_bbox = stage2.merge_bboxes(mrz_lines)
    mrz_crop = stage2.crop_mrz(aligned, mrz_bbox)
    stage2.save(mrz_crop, "mrz_region.png")

    # Step 8 — Debug visualisation
    debug_img = stage2.draw_debug_boxes(aligned, mrz_lines, mrz_bbox)
    stage2.save(debug_img, "mrz_detected.png")

    print("\n[Stage 2] Complete.\n")

    # ======================================================================
    # STAGE 3 — MRZ strip → cleaned image → OCR text
    # ======================================================================
    print("=" * 60)
    print("STAGE 3 — MRZ OCR")
    print("=" * 60)

    # mrz_crop is BGR from Stage 2; pass directly to clean_mrz_image
    # (it handles upscale + grayscale + threshold internally)

    # Step 1 — Clean (upscale → gray → blur → Otsu)
    mrz_clean = stage3.clean_mrz_image(mrz_crop)
    stage3.save(mrz_clean, "mrz_clean.png")

    # Step 2 — Tesseract OCR
    mrz_text  = stage3.run_ocr(mrz_clean)

    print("\n[Stage 3] Final MRZ:")
    print("-" * 60)
    print(mrz_text if mrz_text else "(no text detected)")
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
    print(mrz_text if mrz_text else "(no text)")


if __name__ == "__main__":
    main()
