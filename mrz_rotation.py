import cv2
import detect_mrz as stage2


def _with_prefix(filename: str, prefix: str | None) -> str:
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


def detect_mrz_with_rotation_fallback(aligned_bgr, output_prefix: str | None = None):
    """
    Try MRZ detection on current image and common rotations.

    Returns (working_image, mrz_lines, mrz_bbox, used_label) or None.
    """
    attempts = [
        ("original", None),
        ("rot90_clockwise", cv2.ROTATE_90_CLOCKWISE),
        ("rot90_counterclockwise", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("rot180", cv2.ROTATE_180),
    ]

    for label, rotate_code in attempts:
        working = aligned_bgr if rotate_code is None else cv2.rotate(aligned_bgr, rotate_code)

        img_h, img_w = working.shape[:2]
        img_gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        print(f"[Stage 2] Trying orientation: {label} ({img_w}x{img_h})")

        roi_top = int(img_h * stage2.ROI_TOP_FRACTION)
        roi = img_gray[roi_top:, :]

        blackhat = stage2.apply_blackhat(roi)
        gradient = stage2.compute_gradient(blackhat)
        thresh = stage2.threshold_image(gradient)
        closed = stage2.close_and_dilate(thresh)

        mrz_lines, _ = stage2.find_mrz_contours(closed, img_h, img_w, roi_y_offset=roi_top)
        if mrz_lines:
            stage2.save(blackhat, _with_prefix("mrz_blackhat.png", output_prefix))
            stage2.save(gradient, _with_prefix("mrz_gradient.png", output_prefix))
            stage2.save(thresh, _with_prefix("mrz_threshold.png", output_prefix))
            stage2.save(closed, _with_prefix("mrz_closed.png", output_prefix))
            return working, mrz_lines, stage2.merge_bboxes(mrz_lines), label

    return None
