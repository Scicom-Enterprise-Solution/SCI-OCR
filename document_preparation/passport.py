"""
Passport MRZ OCR Preprocessing Pipeline — Stage 1
PDF → Render Image → Detect Document Contour → Perspective Correction
"""

import os
import sys
import cv2
import numpy as np
from env_utils import load_env_file
from document_inputs.pdf_input import render_pdf_page
from logger_utils import is_debug_enabled
from path_utils import to_repo_relative


load_env_file()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_PATH = os.getenv("PDF_PATH", "passport.pdf")   # Input PDF (override via CLI arg)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DPI = 300
ALIGNED_MAX_DIM = int(os.getenv("ALIGNED_MAX_DIM", "1200"))

# Canny thresholds — adjust if edges are missed or noisy
CANNY_LOW = 30
CANNY_HIGH = 100

# Gaussian blur kernel size (must be odd)
BLUR_KERNEL = (5, 5)

# Contour approximation epsilon factor (fraction of arc length)
APPROX_EPSILON = 0.02

# Downscale longest image dimension to this size for contour detection.
# Working at full 300 DPI resolution fires Canny on internal passport details
# instead of the large-scale document boundary.
DETECTION_MAX_DIM = 1000

# Morphological close kernel applied after Canny to bridge boundary gaps
CLOSE_KERNEL_SIZE = 15

# Minimum contour area as a fraction of the detection image area
MIN_AREA_FRACTION = 0.05

# Progressive epsilon factors for contour approximation fallback.
# Some photos produce noisy boundaries where 0.02 is too strict.
APPROX_EPSILON_FACTORS = (0.02, 0.03, 0.04, 0.05, 0.08)

# Acceptable aspect-ratio range for the document rectangle candidate.
# Covers both landscape and portrait captures with perspective skew.
MIN_DOC_ASPECT_RATIO = 0.45
MAX_DOC_ASPECT_RATIO = 2.40
ESSENTIAL_OUTPUTS = {"aligned_passport.png"}


# ---------------------------------------------------------------------------
# Step 2 — Grayscale, blur, Canny edge detection
# ---------------------------------------------------------------------------

def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert to grayscale, downscale for detection, apply Gaussian blur,
    run Canny edge detection, and close gaps with morphological closing.

    Working at full 300 DPI would fire Canny on passport content details,
    yielding hundreds of tiny contours and missing the document boundary.
    Downscaling keeps only large-scale structure, making the outer edge dominant.

    Returns (edges image at detection scale, detection_scale factor).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # --- Downscale so the longest dimension = DETECTION_MAX_DIM
    detection_scale = min(1.0, DETECTION_MAX_DIM / max(h, w))
    det_w = int(w * detection_scale)
    det_h = int(h * detection_scale)
    small = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_AREA)
    print(f"[Step 2] Detection scale: {detection_scale:.3f} -> {det_w}x{det_h} px")

    # Gaussian blur reduces noise so Canny picks up real edges, not texture
    blurred = cv2.GaussianBlur(small, BLUR_KERNEL, sigmaX=0)

    # Canny: two-stage edge detection using hysteresis thresholds
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Morphological close: dilates then erodes to bridge short gaps in the
    # document boundary that Canny would otherwise leave open.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    print(f"[Step 2] Edge detection complete  "
          f"(Canny: {CANNY_LOW}/{CANNY_HIGH}, morph-close kernel: {CLOSE_KERNEL_SIZE}px)")
    return edges_closed, detection_scale


# ---------------------------------------------------------------------------
# Step 3 — Find the largest quadrilateral contour (document boundary)
# ---------------------------------------------------------------------------

def find_document_contour(
    edges: np.ndarray,
    img_bgr: np.ndarray,
    detection_scale: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Detect contours in the (downscaled) edge map and return the 4-vertex
    contour that most likely represents the passport/document boundary.

    The found corner points are scaled back to the original image coordinates
    so the perspective warp operates on the full-resolution image.

    Returns (4-point array in original coords or None,
             annotated BGR image for debugging at original resolution).
    """
    det_h, det_w = edges.shape[:2]
    min_area = det_h * det_w * MIN_AREA_FRACTION

    # RETR_LIST: retrieve all contours (EXTERNAL can miss the boundary if the
    # document edge forms a partially-open contour after morphological close).
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort largest area first — the document should dominate the scan
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(f"[Step 3] Found {len(contours)} contour(s) in {det_w}×{det_h} detection image.")
    print(f"[Step 3] Top-5 areas: {[int(cv2.contourArea(c)) for c in contours[:5]]}")
    print(f"[Step 3] Min area threshold: {int(min_area)} ({MIN_AREA_FRACTION*100:.0f}% of detection image)")

    # Debug image is drawn at full original resolution (scale contours up)
    debug_img = img_bgr.copy()
    inv_scale = 1.0 / detection_scale
    quad_pts_full = None
    best_fallback_pts = None
    best_fallback_score = -1.0

    def quad_score(quad_pts_det: np.ndarray, area_det: float) -> float:
        """Score candidate quads by area, centrality, and plausible aspect ratio."""
        x, y, w, h = cv2.boundingRect(quad_pts_det.astype(np.int32))
        if w <= 0 or h <= 0:
            return -1.0

        bbox_area = float(w * h)
        fill_ratio = min(1.0, max(0.0, area_det / max(1.0, bbox_area)))

        aspect = w / float(h)
        if not (MIN_DOC_ASPECT_RATIO <= aspect <= MAX_DOC_ASPECT_RATIO):
            return -1.0

        cx = x + w / 2.0
        cy = y + h / 2.0
        dx = abs(cx - det_w / 2.0) / max(1.0, det_w / 2.0)
        dy = abs(cy - det_h / 2.0) / max(1.0, det_h / 2.0)
        center_penalty = min(1.0, (dx + dy) / 2.0)

        area_norm = min(1.0, area_det / max(1.0, det_w * det_h))
        return area_norm * 0.70 + fill_ratio * 0.25 + (1.0 - center_penalty) * 0.05

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Reject contours that are too small to be the document boundary
        if area < min_area:
            print(f"[Step 3] Contour #{idx} area={int(area)} — below threshold, stopping search.")
            break

        perimeter = cv2.arcLength(contour, closed=True)

        approx = None
        for eps_factor in APPROX_EPSILON_FACTORS:
            eps = eps_factor * perimeter
            candidate = cv2.approxPolyDP(contour, eps, closed=True)
            if len(candidate) == 4:
                approx = candidate
                print(f"[Step 3] Contour #{idx}: area={int(area):,}  vertices=4  (eps={eps_factor:.2f})")
                break

        if approx is not None:
            quad_pts_det = approx.reshape(4, 2).astype(np.float32)
            score = quad_score(quad_pts_det, area)
            if score > best_fallback_score:
                best_fallback_score = score
                best_fallback_pts = quad_pts_det
            print(f"[Step 3]   candidate quad score={score:.3f}")
        else:
            # Fallback: minimum-area rectangle often succeeds when contour has many vertices.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            score = quad_score(box, area)
            print(f"[Step 3] Contour #{idx}: area={int(area):,}  vertices>4  minRect score={score:.3f}")
            if score > best_fallback_score:
                best_fallback_score = score
                best_fallback_pts = box

            # Draw rejected/inspected contour approximation in blue for diagnostics.
            contour_full = (contour.astype(np.float32) * inv_scale).astype(np.int32)
            cv2.drawContours(debug_img, [contour_full], -1, (255, 0, 0), thickness=2)

    if best_fallback_pts is not None and best_fallback_score > 0:
        quad_pts_full = best_fallback_pts * inv_scale
        print(f"[Step 3] [OK] Selected best quadrilateral candidate with score={best_fallback_score:.3f}")
        print(f"[Step 3]   Detection corners: {best_fallback_pts.tolist()}")
        print(f"[Step 3]   Full-res corners:  {quad_pts_full.tolist()}")

        quad_cnt_full = (best_fallback_pts.reshape(-1, 1, 2) * inv_scale).astype(np.int32)
        cv2.drawContours(debug_img, [quad_cnt_full], -1, (0, 255, 0), thickness=6)
        for pt in quad_pts_full:
            cv2.circle(debug_img, tuple(pt.astype(int)), radius=20,
                       color=(0, 0, 255), thickness=-1)

    if quad_pts_full is None:
        print("[Step 3] [FAIL] No quadrilateral contour detected. "
              "Check output/edges.png — try lowering CANNY thresholds, "
              "increasing CLOSE_KERNEL_SIZE, or decreasing MIN_AREA_FRACTION.")

    return quad_pts_full, debug_img


# ---------------------------------------------------------------------------
# Step 4 — Perspective correction (bird's-eye warp)
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Arrange 4 points in the canonical order:
    [top-left, top-right, bottom-right, bottom-left].

    Strategy:
    - Top-left  → smallest (x + y)
    - Bottom-right → largest  (x + y)
    - Top-right → smallest (y - x)   (or largest x - y)
    - Bottom-left → largest  (y - x)
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)               # x + y
    diff = np.diff(pts, axis=1)       # y - x

    ordered[0] = pts[np.argmin(s)]    # top-left
    ordered[2] = pts[np.argmax(s)]    # bottom-right
    ordered[1] = pts[np.argmin(diff)] # top-right
    ordered[3] = pts[np.argmax(diff)] # bottom-left

    return ordered


def perspective_correction(
    img_bgr: np.ndarray,
    quad_pts: np.ndarray,
) -> np.ndarray:
    """
    Warp the detected quadrilateral to an axis-aligned rectangle, preserving
    the natural aspect ratio of the detected document.
    """
    ordered = order_points(quad_pts)
    tl, tr, br, bl = ordered

    print(f"[Step 4] Ordered corners -> "
          f"TL={tl.tolist()}  TR={tr.tolist()}  BR={br.tolist()}  BL={bl.tolist()}")

    # Compute output width: max of top-edge and bottom-edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    out_w = int(max(width_top, width_bot))

    # Compute output height: max of left-edge and right-edge lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_h = int(max(height_left, height_right))

    print(f"[Step 4] Output dimensions: {out_w}x{out_h} px (W x H)")

    # Destination rectangle corners (top-left origin)
    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    # Compute the 3×3 perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered, dst)

    # Apply the warp
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))

    return warped


def resize_aligned_image(img_bgr: np.ndarray, max_dim: int = ALIGNED_MAX_DIM) -> tuple[np.ndarray, dict]:
    """
    Cap the aligned working image to a predictable maximum dimension for
    Stage 2/3. This preserves aspect ratio and avoids pushing oversized
    source images through MRZ detection and OCR at full original resolution.
    """
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    meta = {
        "original_width": int(w),
        "original_height": int(h),
        "resized": False,
        "scale": 1.0,
        "max_dim": int(max_dim),
        "width": int(w),
        "height": int(h),
    }

    if max_dim <= 0 or longest <= max_dim:
        return img_bgr, meta

    scale = max_dim / float(longest)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    meta.update({
        "resized": True,
        "scale": round(scale, 6),
        "width": int(target_w),
        "height": int(target_h),
    })
    print(
        f"[Step 4] Aligned image capped to {target_w}x{target_h} px "
        f"(scale={scale:.3f}, max_dim={max_dim})"
    )
    return resized, meta


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save(img: np.ndarray, filename: str) -> None:
    """Save an image to the output directory and print a confirmation."""
    if not is_debug_enabled() and filename not in ESSENTIAL_OUTPUTS:
        return
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[Save]  {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow overriding the PDF path via the first CLI argument
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH

    # Create output directory (safe if it already exists)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Render PDF → numpy image
    # ------------------------------------------------------------------
    img_bgr = render_pdf_page(pdf_path, dpi=DPI)
    save(img_bgr, "rendered_page.png")

    # ------------------------------------------------------------------
    # Step 2 — Edge detection (downscaled for robust contour detection)
    # ------------------------------------------------------------------
    edges, detection_scale = preprocess_image(img_bgr)
    # Save a visually useful version: upscale edges back to original size
    edges_display = cv2.resize(
        edges,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    save(edges_display, "edges.png")

    # ------------------------------------------------------------------
    # Step 3 — Contour detection
    # ------------------------------------------------------------------
    quad_pts, contour_debug = find_document_contour(edges, img_bgr, detection_scale)
    save(contour_debug, "contours.png")

    if quad_pts is None:
        print("\n[Pipeline] Aborting: could not locate a document boundary.")
        print("           Check output/edges.png and output/contours.png for clues.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4 — Perspective correction
    # ------------------------------------------------------------------
    aligned = perspective_correction(img_bgr, quad_pts)
    aligned, _ = resize_aligned_image(aligned)
    save(aligned, "aligned_passport.png")

    print("\n[Pipeline] Done. All outputs written to:", to_repo_relative(OUTPUT_DIR))


if __name__ == "__main__":
    main()
