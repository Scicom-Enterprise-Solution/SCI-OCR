"""
Passport MRZ OCR Preprocessing Pipeline — Stage 2
Aligned passport image → Detect MRZ region → Crop MRZ strip

Input:  output/aligned_passport.png  (produced by document_preparation/passport.py)
Output: output/mrz_region.png        (cropped MRZ strip, ready for OCR)
        output/mrz_detected.png      (annotated full passport with bounding box)
        output/mrz_blackhat.png      (debug: blackhat morphology)
        output/mrz_gradient.png      (debug: horizontal gradient)
        output/mrz_threshold.png     (debug: Otsu threshold)
        output/mrz_closed.png        (debug: morphological closing)
"""

import os
import sys
import cv2
import numpy as np
from env_utils import load_env_file
from logger_utils import is_debug_enabled
from path_utils import to_repo_relative


load_env_file()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "output")
INPUT_IMAGE = os.path.join(OUTPUT_DIR, "aligned_passport.png")

# Blackhat kernel — wide enough to span an MRZ character, short enough to
# stay within a single text line.  Typical OCR-B character is ~30-40px wide
# at 300 DPI; 13 px wide captures the dense horizontal structure well.
BLACKHAT_KERNEL_W = 13
BLACKHAT_KERNEL_H = 5

# Closing kernel — joins individual characters into a solid horizontal band
CLOSE_KERNEL_W = 21
CLOSE_KERNEL_H = 21

# Extra dilation iterations after closing to fatten the band vertically
DILATE_ITERATIONS = 3

# ROI: restrict MRZ detection to the bottom portion of the passport.
# The MRZ always sits in the lower ~35% of the biographical data page;
# skipping the top avoids spurious wide contours from the photo / text fields.
ROI_TOP_FRACTION = 0.65   # top of ROI as fraction of full image height

# MRZ candidate filter constraints
MIN_ASPECT_RATIO       = 5.0    # width / height — MRZ lines are very elongated
MIN_WIDTH_FRACTION     = 0.50   # bounding box width > 50% of passport width
MAX_HEIGHT_FRACTION    = 0.15   # bounding box height < 15% of passport height
LOWER_HALF_FRACTION    = 0.45   # top of bounding box must be below this y fraction
MIN_MERGED_HEIGHT_FRACTION = 0.06  # merged TD3 MRZ band should not be implausibly thin

# Number of MRZ lines to collect (standard passport MRZ = 2 lines)
MRZ_LINE_COUNT = 2

# Padding added around the final MRZ crop (pixels)
CROP_PAD_X = 10
CROP_PAD_Y = 15   # generous vertical padding to capture full character height
ESSENTIAL_OUTPUTS = {"mrz_region.png", "mrz_detected.png"}

# Scale-aware detection: very short ROIs need to be enlarged before the fixed
# morphology kernels become effective on the MRZ line structure.
DETECTION_MIN_ROI_HEIGHT = 216
DETECTION_MAX_SMALL_IMAGE_WIDTH = 600
DETECTION_MAX_UPSCALE = 2.5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save(img: np.ndarray, filename: str) -> None:
    """Save an image to OUTPUT_DIR and print a confirmation."""
    if not is_debug_enabled() and filename not in ESSENTIAL_OUTPUTS:
        return
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[Save]  {path}")


def prepare_detection_roi(gray: np.ndarray) -> tuple[np.ndarray, int, float]:
    """
    Build the bottom-of-page ROI used for MRZ detection and upscale it when the
    source image is small enough that fixed morphology kernels would otherwise
    overrun the text structure.
    """
    img_h, img_w = gray.shape[:2]
    roi_top = int(img_h * ROI_TOP_FRACTION)
    roi = gray[roi_top:, :]
    roi_h = max(1, roi.shape[0])

    upscale = 1.0
    if img_w <= DETECTION_MAX_SMALL_IMAGE_WIDTH and roi_h < DETECTION_MIN_ROI_HEIGHT:
        upscale = max(upscale, DETECTION_MIN_ROI_HEIGHT / float(roi_h))
    upscale = min(DETECTION_MAX_UPSCALE, upscale)

    if upscale > 1.0:
        scaled_w = max(1, int(round(img_w * upscale)))
        scaled_h = max(1, int(round(roi_h * upscale)))
        roi = cv2.resize(roi, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        print(f"[Step 1] Detection ROI upscaled x{upscale:.2f} -> {scaled_w}x{scaled_h} px")

    return roi, roi_top, upscale


def scale_bboxes_back(
    bboxes: list[tuple[int, int, int, int]],
    upscale: float,
) -> list[tuple[int, int, int, int]]:
    """Map detection-space boxes back to original image coordinates."""
    if upscale <= 1.0:
        return bboxes

    scaled = []
    for x, y, w, h in bboxes:
        scaled.append((
            int(round(x / upscale)),
            int(round(y / upscale)),
            max(1, int(round(w / upscale))),
            max(1, int(round(h / upscale))),
        ))
    return scaled


def detect_mrz_lines(
    img_bgr: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], dict[str, np.ndarray]]:
    """
    Run the Stage 2 MRZ detection pipeline on an image and return line boxes in
    original-image coordinates plus intermediate debug images.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi, roi_top, upscale = prepare_detection_roi(img_gray)

    blackhat = apply_blackhat(roi)
    gradient = compute_gradient(blackhat)
    thresh = threshold_image(gradient)
    closed = close_and_dilate(thresh)

    detect_img_h = int(round(img_h * upscale))
    detect_img_w = int(round(img_w * upscale))
    detect_roi_top = int(round(roi_top * upscale))
    mrz_lines, debug_records = find_mrz_contours(
        closed,
        detect_img_h,
        detect_img_w,
        roi_y_offset=detect_roi_top,
    )
    mrz_lines = scale_bboxes_back(mrz_lines, upscale)

    debug_images = {
        "blackhat": blackhat,
        "gradient": gradient,
        "thresh": thresh,
        "closed": closed,
    }
    return mrz_lines, {"images": debug_images, "debug_records": debug_records, "upscale": upscale}


# ---------------------------------------------------------------------------
# Step 1 — Load image and convert to grayscale
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the aligned passport image (BGR) and return both the colour and
    grayscale versions.
    """
    if not os.path.isfile(path):
        print(f"[ERROR] Input image not found: {path}")
        print("        Run document_preparation/passport.py first to produce aligned_passport.png")
        sys.exit(1)

    img_bgr = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = img_bgr.shape[:2]
    print(f"[Step 1] Loaded '{path}' -> {w}x{h} px (W x H)")
    return img_bgr, img_gray


# ---------------------------------------------------------------------------
# Step 2 — Blackhat morphology
# ---------------------------------------------------------------------------

def apply_blackhat(gray: np.ndarray) -> np.ndarray:
    """
    Blackhat = closing(image) − image.

    It highlights dark objects (ink) against a bright background (paper).
    The wide, flat kernel responds strongly to horizontal bands of dense
    dark marks — exactly what the MRZ looks like.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (BLACKHAT_KERNEL_W, BLACKHAT_KERNEL_H)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    print(f"[Step 2] Blackhat kernel: {BLACKHAT_KERNEL_W}x{BLACKHAT_KERNEL_H}")
    return blackhat


# ---------------------------------------------------------------------------
# Step 3 — Horizontal gradient (Sobel X)
# ---------------------------------------------------------------------------

def compute_gradient(blackhat: np.ndarray) -> np.ndarray:
    """
    A horizontal Sobel gradient amplifies sharp transitions in the X direction.
    Because MRZ text consists of many vertical strokes packed tightly, it
    produces a very high horizontal gradient energy compared to the rest of the
    passport page.

    Output is normalised to uint8 [0, 255].
    """
    # Sobel in X direction; ddepth=cv2.CV_32F keeps sign information
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad = np.absolute(grad)

    # Normalise to 0-255 for thresholding and display
    grad_norm = np.zeros_like(grad, dtype=np.uint8)
    cv2.normalize(grad, grad_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                  dtype=cv2.CV_8U)

    print("[Step 3] Horizontal gradient computed and normalised")
    return grad_norm


# ---------------------------------------------------------------------------
# Step 4 — Gaussian blur + Otsu threshold
# ---------------------------------------------------------------------------

def threshold_image(gradient: np.ndarray) -> np.ndarray:
    """
    Blur first to merge adjacent gradient peaks within a single character,
    then use Otsu's method to automatically separate text from background.
    """
    # Blur with a wide kernel to smear nearby gradient peaks into solid blobs
    blurred = cv2.GaussianBlur(gradient, (21, 21), sigmaX=0)

    # Otsu automatically computes the optimal threshold from the histogram
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    print("[Step 4] Threshold (Otsu) applied")
    return thresh


# ---------------------------------------------------------------------------
# Step 5 — Morphological closing + dilation
# ---------------------------------------------------------------------------

def close_and_dilate(thresh: np.ndarray) -> np.ndarray:
    """
    Closing = dilation then erosion.  With a wide rectangular kernel this
    bridges the gaps between individual MRZ characters, turning two rows of
    text into two solid horizontal bars.  Additional dilation then merges
    those two bars into one thick band, making it easy to detect as one
    large contour.
    """
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_W, CLOSE_KERNEL_H)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # Extra dilation iterations to vertically fatten the band
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=DILATE_ITERATIONS)

    print(f"[Step 5] Morphological closing ({CLOSE_KERNEL_W}x{CLOSE_KERNEL_H}) "
          f"+ {DILATE_ITERATIONS} dilation iteration(s)")
    return dilated


# ---------------------------------------------------------------------------
# Step 6 — Contour detection and MRZ candidate filtering
# ---------------------------------------------------------------------------

def find_mrz_contours(
    closed: np.ndarray,
    img_h: int,
    img_w: int,
    roi_y_offset: int = 0,
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """
    Find all contours in the processed binary image and collect up to
    MRZ_LINE_COUNT candidates that satisfy:
      - width  > MIN_WIDTH_FRACTION  of passport width  (MRZ spans full width)
      - aspect ratio (w/h) > MIN_ASPECT_RATIO           (very elongated)
      - height < MAX_HEIGHT_FRACTION of passport height (thin strip)
      - top edge is in the lower half (LOWER_HALF_FRACTION)

    Candidates are sorted by Y coordinate and the MRZ_LINE_COUNT lowest
    contours are returned — these correspond to the two MRZ text lines,
    which always sit at the very bottom of the biographical data page.

    Returns (list of up to MRZ_LINE_COUNT bounding rects, debug records).
    """
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print(f"[Step 6] {len(contours)} contours found after morphology")

    candidates = []
    debug_records = []

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area   = cv2.contourArea(cnt)
        aspect = w / float(h) if h > 0 else 0.0

        # Translate ROI-relative Y back to full-image coordinates so that
        # all fraction checks are consistent with the original image size.
        y_orig = y + roi_y_offset

        rec = {
            "idx": idx, "x": x, "y": y_orig, "w": w, "h": h,
            "area": area, "aspect": aspect,
        }

        w_frac = w / img_w
        h_frac = h / img_h
        y_frac = y_orig / img_h   # top edge as fraction of full image height

        # Log every significant contour
        if area > 100:
            print(f"[Step 6]   #{idx:>3}  area={int(area):>8,}  "
                  f"bbox=({x},{y_orig},{w},{h})  aspect={aspect:.2f}  "
                  f"w%={w_frac*100:.1f}  y%={y_frac*100:.1f}")

        passes = (
            w_frac >= MIN_WIDTH_FRACTION  and   # wide enough to be MRZ
            h_frac <= MAX_HEIGHT_FRACTION and   # not too tall
            aspect >= MIN_ASPECT_RATIO    and   # horizontally elongated
            y_frac >= LOWER_HALF_FRACTION        # in the lower half
        )

        rec["passes"] = passes
        debug_records.append(rec)

        if passes:
            candidates.append((x, y_orig, w, h))  # Y in original image coords

    if not candidates:
        print("[Step 6] [FAIL] No MRZ candidates passed all filters.")
        return [], debug_records

    # Sort all candidates top-to-bottom by their Y coordinate.
    candidates.sort(key=lambda r: r[1])
    print(f"[Step 6] {len(candidates)} candidate(s) passed filters.")

    if len(candidates) == 1:
        # Only one line found — return it as-is
        selected = candidates
    else:
        # Find the pair of candidates that are closest to each other
        # vertically (minimum gap between bottom of upper box and top of
        # lower box).  The two MRZ text lines are always adjacent so they
        # will have the smallest gap of any qualifying pair, making this
        # robust against other wide contours that may pass the width/aspect
        # filters but sit far away from the MRZ strip.
        min_gap = float("inf")
        best_i, best_j = 0, 1

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                # Bottom of the upper box
                y_bottom_i = candidates[i][1] + candidates[i][3]
                # Top of the lower box
                y_top_j    = candidates[j][1]
                gap = abs(y_top_j - y_bottom_i)

                print(f"[Step 6]   pair ({i},{j})  gap={gap}px")

                if gap < min_gap:
                    min_gap  = gap
                    best_i, best_j = i, j

        selected = [candidates[best_i], candidates[best_j]]
        selected.sort(key=lambda r: r[1])   # ensure top-to-bottom order
        print(f"[Step 6] Closest pair: indices ({best_i},{best_j})  "
              f"vertical gap={min_gap}px")

    for i, bbox in enumerate(selected):
        print(f"[Step 6] [OK] MRZ line {i+1}/{len(selected)}: bbox={bbox}")

    return selected, debug_records


def merge_bboxes(
    bboxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    """
    Compute the tightest axis-aligned rectangle that contains all supplied
    bounding boxes.
    """
    x1 = min(b[0]          for b in bboxes)
    y1 = min(b[1]          for b in bboxes)
    x2 = max(b[0] + b[2]   for b in bboxes)   # x + w
    y2 = max(b[1] + b[3]   for b in bboxes)   # y + h
    merged = (x1, y1, x2 - x1, y2 - y1)
    print(f"[Step 6] Merged bbox of {len(bboxes)} line(s): {merged}")
    return merged


def is_plausible_merged_mrz_bbox(
    bbox: tuple[int, int, int, int],
    img_h: int,
) -> bool:
    _, _, _, h = bbox
    min_height = max(1, int(round(img_h * MIN_MERGED_HEIGHT_FRACTION)))
    if h < min_height:
        print(
            f"[Step 6] [FAIL] Rejected merged MRZ bbox as too thin: "
            f"h={h}px < min={min_height}px ({MIN_MERGED_HEIGHT_FRACTION:.2%} of image height)"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Step 7 — Crop MRZ region
# ---------------------------------------------------------------------------

def crop_mrz(img_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop the MRZ strip from the full passport image, adding a small padding
    so that characters at the edges are not clipped.
    """
    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]

    # Clamp padded coordinates to image bounds
    x1 = max(0, x - CROP_PAD_X)
    y1 = max(0, y - CROP_PAD_Y)
    x2 = min(img_w, x + w + CROP_PAD_X)
    y2 = min(img_h, y + h + CROP_PAD_Y)

    crop = img_bgr[y1:y2, x1:x2]
    print(f"[Step 7] MRZ crop: ({x1},{y1}) -> ({x2},{y2})  "
          f"size={x2-x1}x{y2-y1} px")
    return crop


# ---------------------------------------------------------------------------
# Step 8 — Draw bounding box on full image for debugging
# ---------------------------------------------------------------------------

def draw_debug_boxes(
    img_bgr: np.ndarray,
    line_bboxes: list[tuple[int, int, int, int]],
    merged_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Draw individual MRZ line bounding boxes (yellow) and the merged bounding
    box (green) with a label on a copy of the passport image.
    """
    debug = img_bgr.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness  = 2

    # Individual line boxes in yellow
    for i, (x, y, w, h) in enumerate(line_bboxes):
        cv2.rectangle(debug, (x, y), (x + w, y + h),
                      color=(0, 215, 255), thickness=2)
        cv2.putText(debug, f"Line {i+1}", (x + 4, y - 8),
                    font, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

    # Merged (combined) box in green
    mx, my, mw, mh = merged_bbox
    cv2.rectangle(debug, (mx, my), (mx + mw, my + mh),
                  color=(0, 255, 0), thickness=3)

    # Label above the merged box
    label = "MRZ Region"
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = mx
    label_y = max(my - 10, th + baseline)

    cv2.rectangle(debug,
                  (label_x, label_y - th - baseline),
                  (label_x + tw, label_y + baseline),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(debug, label, (label_x, label_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return debug


# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow overriding input image via CLI argument
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_IMAGE

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------
    img_bgr, img_gray = load_image(input_path)
    img_h, img_w = img_bgr.shape[:2]

    # Restrict processing to the bottom portion where MRZ always appears.
    # Working on the full page exposes the pipeline to portrait text and
    # biographical fields that create spurious wide contours.
    roi_top = int(img_h * ROI_TOP_FRACTION)
    print(f"[Step 1] ROI: bottom {100*(1-ROI_TOP_FRACTION):.0f}% of image  "
            f"(rows {roi_top}-{img_h},  {img_w}x{img_h - roi_top} px)")
    # Step 2 — Blackhat morphology (on ROI)
    # ------------------------------------------------------------------
    mrz_lines, detection_meta = detect_mrz_lines(img_bgr)
    blackhat = detection_meta["images"]["blackhat"]
    save(blackhat, "mrz_blackhat.png")

    # ------------------------------------------------------------------
    # Step 3 — Horizontal gradient
    # ------------------------------------------------------------------
    gradient = detection_meta["images"]["gradient"]
    save(gradient, "mrz_gradient.png")

    # ------------------------------------------------------------------
    # Step 4 — Threshold
    # ------------------------------------------------------------------
    thresh = detection_meta["images"]["thresh"]
    save(thresh, "mrz_threshold.png")

    # ------------------------------------------------------------------
    # Step 5 — Morphological closing + dilation
    # ------------------------------------------------------------------
    closed = detection_meta["images"]["closed"]
    save(closed, "mrz_closed.png")

    if not mrz_lines:
        print("\n[Pipeline] Aborting: MRZ region not detected.")
        print("           Inspect the debug images in output/ for clues:")
        print("            mrz_blackhat.png  — should brighten the MRZ text area")
        print("            mrz_gradient.png  — should show a bright band at the bottom")
        print("            mrz_threshold.png — should show a thick bar at the bottom")
        print("            mrz_closed.png    — should show one solid rectangle near bottom")
        print("           Try reducing MIN_WIDTH_FRACTION, MIN_ASPECT_RATIO, or LOWER_HALF_FRACTION.")
        sys.exit(1)

    # Merge the individual line bboxes into one combined rectangle
    mrz_bbox = merge_bboxes(mrz_lines)
    if not is_plausible_merged_mrz_bbox(mrz_bbox, img_bgr.shape[0]):
        print("\n[Pipeline] Aborting: merged MRZ region is not structurally plausible.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7 — Crop MRZ
    # ------------------------------------------------------------------
    mrz_crop = crop_mrz(img_bgr, mrz_bbox)
    save(mrz_crop, "mrz_region.png")

    # ------------------------------------------------------------------
    # Step 8 — Debug visualisation
    # ------------------------------------------------------------------
    debug_img = draw_debug_boxes(img_bgr, mrz_lines, mrz_bbox)
    save(debug_img, "mrz_detected.png")

    print(f"\n[Pipeline] Done. MRZ region saved to: "
          f"{to_repo_relative(os.path.join(OUTPUT_DIR, 'mrz_region.png'))}")


if __name__ == "__main__":
    main()
