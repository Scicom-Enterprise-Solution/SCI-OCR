import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


FAST_OCR_VARIANT_SOURCES = ("gray", "clahe")
FAST_OCR_VARIANT_SCALES = (2,)
FAST_OCR_VARIANT_THRESHOLDS = ("otsu",)
FAST_OCR_SPLIT_LABELS = ("projection", "half")
PADDLE_FAST_VARIANT_SOURCES = ("gray", "clahe")
PADDLE_FAST_VARIANT_SCALES = (2,)
PADDLE_FAST_VARIANT_THRESHOLDS = ("otsu", "adaptive")
PADDLE_FAST_SPLIT_LABELS = ("projection", "half", "projection_p4")


def clean_mrz_image(mrz_img):
    if len(mrz_img.shape) == 3:
        img = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
    else:
        img = mrz_img.copy()

    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_thresh(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def _otsu_thresh(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _thicken(binary):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def _resize(img, scale: int):
    if scale == 1:
        return img.copy()
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _resize_to_height(img, target_height: int):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or h == target_height:
        return img.copy()
    target_width = max(1, int(round(w * (target_height / float(h)))))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def _cap_width(img, max_width: int):
    h, w = img.shape[:2]
    if w <= max_width or h <= 0 or w <= 0:
        return img.copy()
    target_height = max(1, int(round(h * (max_width / float(w)))))
    return cv2.resize(img, (max_width, target_height), interpolation=cv2.INTER_AREA)


def ocr_search_profile(fast_ocr: bool, *, paddle_fast: bool = False, ocr_backend: str = "") -> dict:
    if paddle_fast and ocr_backend == "paddle":
        return {
            "variant_sources": PADDLE_FAST_VARIANT_SOURCES,
            "variant_scales": PADDLE_FAST_VARIANT_SCALES,
            "variant_thresholds": PADDLE_FAST_VARIANT_THRESHOLDS,
            "split_labels": PADDLE_FAST_SPLIT_LABELS,
            "profile_name": "paddle_fast",
        }

    if fast_ocr:
        return {
            "variant_sources": FAST_OCR_VARIANT_SOURCES,
            "variant_scales": FAST_OCR_VARIANT_SCALES,
            "variant_thresholds": FAST_OCR_VARIANT_THRESHOLDS,
            "split_labels": FAST_OCR_SPLIT_LABELS,
            "profile_name": "fast",
        }

    return {
        "variant_sources": ("gray", "clahe"),
        "variant_scales": (2, 3, 4),
        "variant_thresholds": ("otsu", "adaptive", "otsu_thick"),
        "split_labels": (
            "projection",
            "half",
            "projection_m4",
            "projection_p4",
            "projection_m8",
            "projection_p8",
        ),
        "profile_name": "exhaustive",
    }


def prepare_variants(gray, prefix: str, profile: dict):
    variants = []

    variant_sources = profile["variant_sources"]
    base_images = {"gray": gray}
    if "clahe" in variant_sources:
        base_images["clahe"] = _apply_clahe(gray)

    tasks = [
        (source_name, scale, threshold_name)
        for source_name in variant_sources
        for scale in profile["variant_scales"]
        for threshold_name in profile["variant_thresholds"]
    ]

    def build_variant(task):
        source_name, scale, threshold_name = task
        base = base_images[source_name]
        scaled = _resize(base, scale)

        if threshold_name == "otsu":
            image = _otsu_thresh(scaled)
            morph = "none"
            threshold = "otsu"
        elif threshold_name == "adaptive":
            image = _adaptive_thresh(scaled)
            morph = "none"
            threshold = "adaptive"
        elif threshold_name == "otsu_thick":
            image = _thicken(_otsu_thresh(scaled))
            morph = "dilate"
            threshold = "otsu"
        else:
            return None

        return {
            "variant_id": f"{prefix}_{source_name}_s{scale}_{threshold_name}",
            "image": image,
            "meta": {
                "scale": scale,
                "source": source_name,
                "threshold": threshold,
                "morph": morph,
            },
        }

    worker_count = int(profile.get("variant_workers", 1))
    if worker_count <= 1 or len(tasks) <= 1:
        iterator = map(build_variant, tasks)
    else:
        max_workers = min(worker_count, len(tasks), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(build_variant, tasks)
            variants.extend(variant for variant in iterator if variant is not None)
        return variants

    variants.extend(variant for variant in iterator if variant is not None)

    return variants


def split_mrz_lines(img, *, save_debug):
    gray = _to_gray(img)

    if gray.shape[0] < 10:
        h = gray.shape[0]
        return gray[:h // 2, :], gray[h // 2:, :], {
            "split_y": h // 2,
            "method": "half",
            "profile_reliable": False,
            "image_height": h,
        }

    inv = 255 - gray
    proj = inv.mean(axis=1)

    h = gray.shape[0]
    low = max(1, int(h * 0.25))
    high = min(h - 2, int(h * 0.75))

    valley_idx = low + int(np.argmin(proj[low:high]))
    left = max(low, valley_idx - 3)
    right = min(high, valley_idx + 4)
    neighborhood = proj[left:right]

    reliability = False
    if len(neighborhood) > 0:
        local_min = float(np.min(neighborhood))
        local_mean = float(np.mean(proj[low:high]))
        reliability = local_min < (local_mean * 0.8)

    split_y = valley_idx if reliability else h // 2

    line1 = gray[:split_y, :]
    line2 = gray[split_y:, :]

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, split_y), (gray.shape[1] - 1, split_y), (0, 255, 0), 2)
    label = f"split_y={split_y} ({'profile' if reliability else 'half'})"
    cv2.putText(
        vis, label, (5, max(18, split_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    save_debug(vis, "mrz_line_split.png")

    proj_vis_h = 120
    proj_norm = proj.copy()
    if proj_norm.max() > 0:
        proj_norm = proj_norm / proj_norm.max()
    proj_vis = np.full((proj_vis_h, gray.shape[0], 3), 255, dtype=np.uint8)
    for x in range(gray.shape[0]):
        y = int((1.0 - proj_norm[x]) * (proj_vis_h - 1))
        cv2.line(proj_vis, (x, proj_vis_h - 1), (x, y), (80, 80, 80), 1)
    cv2.line(proj_vis, (split_y, 0), (split_y, proj_vis_h - 1), (0, 255, 0), 2)
    proj_vis = cv2.rotate(proj_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
    save_debug(proj_vis, "mrz_line_projection.png")

    meta = {
        "split_y": int(split_y),
        "method": "projection_valley" if reliability else "half",
        "profile_reliable": bool(reliability),
        "image_height": int(h),
    }
    return line1, line2, meta


def split_mrz_lines_at(gray, split_y: int):
    h = gray.shape[0]
    split_y = max(8, min(h - 8, int(split_y)))
    line1 = gray[:split_y, :]
    line2 = gray[split_y:, :]
    return line1, line2


def build_split_candidates(gray, base_meta: dict, profile: dict):
    h = gray.shape[0]
    proj_y = int(base_meta.get("split_y", h // 2))
    half_y = h // 2

    candidates = []
    seen = set()

    split_positions = {
        "projection": proj_y,
        "half": half_y,
        "projection_m4": proj_y - 4,
        "projection_p4": proj_y + 4,
        "projection_m8": proj_y - 8,
        "projection_p8": proj_y + 8,
    }

    for label in profile["split_labels"]:
        y = split_positions[label]
        y = max(8, min(h - 8, y))
        if y in seen:
            continue
        seen.add(y)

        top_ratio = y / h
        bottom_ratio = (h - y) / h

        candidates.append({
            "label": label,
            "split_y": y,
            "top_ratio": top_ratio,
            "bottom_ratio": bottom_ratio,
        })

    return candidates


def estimate_ocr_search_space(
    *,
    fast_ocr: bool,
    tesseract_psms: list[int],
    tesseract_oems: list[int],
    paddle_fast: bool = False,
    ocr_backend: str = "",
) -> dict:
    profile = ocr_search_profile(fast_ocr, paddle_fast=paddle_fast, ocr_backend=ocr_backend)
    per_line_variants = (
        len(profile["variant_sources"])
        * len(profile["variant_scales"])
        * len(profile["variant_thresholds"])
    )
    per_line_attempts = per_line_variants * len(tesseract_psms)
    split_count = len(profile["split_labels"])

    return {
        "fast_ocr": fast_ocr,
        "paddle_fast": bool(paddle_fast and ocr_backend == "paddle"),
        "profile_name": profile.get("profile_name", "unknown"),
        "psms": list(tesseract_psms),
        "oems": list(tesseract_oems),
        "split_count": split_count,
        "per_line_variants": per_line_variants,
        "per_line_attempts": per_line_attempts,
        "total_tesseract_calls": split_count * 2 * per_line_attempts,
    }
