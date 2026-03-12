import os
import cv2
import numpy as np


_CASCADE_FILENAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
]

_DETECTION_CONFIGS = [
    (1.10, 5, 40),
    (1.08, 4, 32),
    (1.06, 3, 24),
    (1.04, 2, 18),
]

_UPSCALE_FACTORS = [1.0, 1.25, 1.50]


def _load_face_cascades() -> list[cv2.CascadeClassifier]:
    cascades: list[cv2.CascadeClassifier] = []
    for filename in _CASCADE_FILENAMES:
        path = os.path.join(cv2.data.haarcascades, filename)
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            cascades.append(cascade)

    if not cascades:
        raise RuntimeError("Cannot load any Haar face cascade from cv2.data.haarcascades")
    return cascades


def _prepare_gray_variants(gray: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []

    # Variant 1: histogram equalization (good default for document photos).
    eq = cv2.equalizeHist(gray)
    variants.append(eq)

    # Variant 2: CLAHE for uneven lighting and shadows.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # Variant 3: light denoising before equalization.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(cv2.equalizeHist(blur))

    return variants


def _to_original_bbox(bbox, scale: float):
    x, y, w, h = bbox
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(w / scale)),
        int(round(h / scale)),
    )


def _iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = max(1, aw * ah)
    b_area = max(1, bw * bh)
    union = a_area + b_area - inter
    return inter / float(max(1, union))


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.35):
    if not boxes:
        return []

    # Keep larger boxes first to stabilize selection.
    boxes = sorted(boxes, key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []

    for box in boxes:
        if all(_iou(box, prev) < iou_threshold for prev in kept):
            kept.append(box)
    return kept


def detect_faces_bgr(image_bgr):
    """Return face bounding boxes as (x, y, w, h)."""
    cascades = _load_face_cascades()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants = _prepare_gray_variants(gray)

    img_h, img_w = gray.shape[:2]
    collected: list[tuple[int, int, int, int]] = []

    for prepared in variants:
        for scale in _UPSCALE_FACTORS:
            if scale == 1.0:
                working = prepared
            else:
                new_w = int(round(img_w * scale))
                new_h = int(round(img_h * scale))
                working = cv2.resize(prepared, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            for cascade in cascades:
                for scale_factor, min_neighbors, min_size in _DETECTION_CONFIGS:
                    faces = cascade.detectMultiScale(
                        working,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(min_size, min_size),
                    )

                    for face in faces:
                        x, y, w, h = _to_original_bbox(face, scale)

                        # Filter tiny/invalid boxes that are usually noise.
                        if w < 18 or h < 18:
                            continue
                        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                            continue

                        collected.append((x, y, w, h))

    deduped = _dedupe_boxes(collected, iou_threshold=0.35)

    # Ultra-relaxed final fallback if everything above failed.
    if not deduped:
        fallback = cascades[0].detectMultiScale(
            variants[0],
            scaleFactor=1.03,
            minNeighbors=1,
            minSize=(16, 16),
        )
        deduped = _dedupe_boxes([tuple(map(int, f)) for f in fallback], iou_threshold=0.30)

    # Keep top candidates by area to reduce orientation noise from tiny false positives.
    deduped = sorted(deduped, key=lambda r: r[2] * r[3], reverse=True)
    return deduped[:30]


def choose_best_face(faces, image_shape):
    """
    Prefer larger faces in the upper part of the image.
    Returns (x, y, w, h) or None.
    """
    if not faces:
        return None

    img_h, img_w = image_shape[:2]
    best = None
    best_score = -1.0

    for (x, y, w, h) in faces:
        area_score = (w * h) / float(img_w * img_h)
        center_y = y + h / 2.0
        center_x = x + w / 2.0

        # Prefer plausible portrait-like face shape; strongly penalize odd boxes.
        ratio = w / float(max(1, h))
        ratio_penalty = max(0.2, 1.0 - min(1.0, abs(ratio - 0.85)))

        # Passport portraits are usually in the upper ~70%.
        upper_bonus = 1.0 if center_y <= img_h * 0.70 else 0.25

        # Face area in passport should not be tiny; add mild center preference.
        center_bonus = 1.0 - min(0.7, abs(center_x - (img_w / 2.0)) / max(1.0, img_w))
        score = area_score * upper_bonus * ratio_penalty * center_bonus

        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(w), int(h))

    return best


def extract_face_crop(image_bgr, bbox, pad_ratio: float = 0.20):
    """Extract a padded face crop from bbox=(x,y,w,h)."""
    if bbox is None:
        return None

    x, y, w, h = bbox
    img_h, img_w = image_bgr.shape[:2]

    px = int(w * pad_ratio)
    py = int(h * pad_ratio)

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)

    return image_bgr[y1:y2, x1:x2].copy()


def draw_face_box(image_bgr, bbox):
    """Draw face bounding box and return annotated copy."""
    out = image_bgr.copy()
    if bbox is None:
        return out

    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 3)
    cv2.putText(out, "Face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return out


def orient_with_face_hint(image_bgr):
    """
    Try common orientations and pick the one with strongest face evidence.

    Returns dict with:
      - image: oriented BGR image
      - label: orientation label
      - rotate_code: OpenCV rotate code or None
      - face_bbox: selected face bbox or None
      - faces_count: number of faces detected in selected orientation
    """
    attempts = [
        ("original", None),
        ("rot90_clockwise", cv2.ROTATE_90_CLOCKWISE),
        ("rot90_counterclockwise", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("rot180", cv2.ROTATE_180),
    ]

    best = {
        "score": -1.0,
        "image": image_bgr,
        "label": "original",
        "rotate_code": None,
        "face_bbox": None,
        "faces_count": 0,
    }

    for label, rotate_code in attempts:
        working = image_bgr if rotate_code is None else cv2.rotate(image_bgr, rotate_code)
        faces = detect_faces_bgr(working)
        face_bbox = choose_best_face(faces, working.shape)
        print(f"[Face] Orientation '{label}': candidates={len(faces)}")

        if face_bbox is None:
            score = 0.0
        else:
            x, y, w, h = face_bbox
            area = (w * h) / float(working.shape[0] * working.shape[1])
            center_y = y + h / 2.0
            upper_bonus = 1.0 if center_y <= working.shape[0] * 0.70 else 0.25
            score = area * upper_bonus + 0.002 * min(len(faces), 5)

        if score > best["score"]:
            best = {
                "score": score,
                "image": working,
                "label": label,
                "rotate_code": rotate_code,
                "face_bbox": face_bbox,
                "faces_count": len(faces),
            }

    best.pop("score", None)
    return best
