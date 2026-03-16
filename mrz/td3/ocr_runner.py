import math

import cv2
import numpy as np

from ocr_backends.paddle_backend import (
    best_paddle_text_candidate as paddle_backend_best_text_candidate,
    extract_paddle_lines as paddle_backend_extract_lines,
    get_paddle_ocr_stats as paddle_backend_get_ocr_stats,
    get_paddle_ocr as paddle_backend_get_ocr,
    paddle_ocr_images as paddle_backend_ocr_images,
    reset_paddle_ocr_stats as paddle_backend_reset_ocr_stats,
    resolve_paddle_use_gpu as paddle_backend_resolve_use_gpu,
    trim_line1_spill as paddle_backend_trim_line1_spill,
)
from ocr_backends.tesseract_backend import ocr_image as tesseract_backend_ocr_image


PADDLE_OCR_BACKEND_LABEL = "paddle"
CANDIDATE_SUPPORT_BONUS_SCALE = 3.0
PADDLE_LINE1_SPILL_SELECTION_PENALTY = 18.0
PADDLE_LINE1_SPILL_COLLAPSE_PENALTY = 6.0


def ocr_image(img, cfg: str, *, tesseract_lang: str, mrz_whitelist: str, normalize_mrz):
    return tesseract_backend_ocr_image(
        img,
        cfg,
        lang=tesseract_lang,
        whitelist=mrz_whitelist,
        normalize_mrz=normalize_mrz,
    )


def resolve_ocr_backends(ocr_backend: str) -> list[str]:
    backend = ocr_backend
    if backend in {"tesseract", "paddle"}:
        return [backend]
    if backend in {"auto", "hybrid", "both"}:
        return ["tesseract", "paddle"]
    print(f"[WARN] Unknown OCR_BACKEND='{backend}', falling back to paddle")
    return ["paddle"]


def trim_line1_spill(text: str, normalize_td3_line1) -> str:
    return paddle_backend_trim_line1_spill(text, normalize_td3_line1)


def resolve_paddle_use_gpu(use_gpu: bool) -> bool:
    return paddle_backend_resolve_use_gpu(use_gpu)


def get_paddle_ocr(lang: str, use_gpu: bool):
    return paddle_backend_get_ocr(lang, use_gpu)


def reset_paddle_ocr_stats() -> None:
    paddle_backend_reset_ocr_stats()


def get_paddle_ocr_stats() -> dict:
    return paddle_backend_get_ocr_stats()


def best_paddle_text_candidate(
    texts: list[str],
    line_kind: str,
    *,
    normalize_mrz,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
    trim_line1_spill_func,
) -> str:
    return paddle_backend_best_text_candidate(
        texts,
        line_kind,
        normalize_mrz=normalize_mrz,
        normalize_td3_line2=normalize_td3_line2,
        score_td3_line1=score_td3_line1,
        score_td3_line2=score_td3_line2,
        trim_line1_spill_func=trim_line1_spill_func,
    )


def paddle_ocr_image(
    img,
    *,
    line_kind: str | None,
    paddle_lang: str,
    paddle_use_gpu: bool,
    normalize_mrz,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
    trim_line1_spill_func,
) -> str:
    ocr = get_paddle_ocr(paddle_lang, paddle_use_gpu)
    paddle_img = img

    if isinstance(paddle_img, np.ndarray) and paddle_img.ndim == 2:
        paddle_img = cv2.cvtColor(paddle_img, cv2.COLOR_GRAY2BGR)

    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(paddle_img)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Installed Paddle runtime failed during CPU inference. "
                "This project expects the official PaddleOCR-supported runtime; "
                "reinstall the pinned versions from requirements-paddle.txt in .venv."
            ) from exc
    elif hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(paddle_img, cls=False)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Installed Paddle runtime failed during CPU inference. "
                "This project expects the official PaddleOCR-supported runtime; "
                "reinstall the pinned versions from requirements-paddle.txt in .venv."
            ) from exc
    else:
        raise RuntimeError("Unsupported PaddleOCR instance: missing predict/ocr method")

    texts = paddle_backend_extract_lines(result, normalize_mrz)
    if not texts:
        return ""

    if line_kind in {"line1", "line2"}:
        return best_paddle_text_candidate(
            texts,
            line_kind,
            normalize_mrz=normalize_mrz,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
            trim_line1_spill_func=trim_line1_spill_func,
        )

    return normalize_mrz("".join(texts))


def paddle_ocr_images(
    images,
    *,
    line_kind: str | None,
    paddle_lang: str,
    paddle_use_gpu: bool,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
):
    return paddle_backend_ocr_images(
        images,
        line_kind=line_kind,
        paddle_lang=paddle_lang,
        paddle_use_gpu=paddle_use_gpu,
        normalize_mrz=normalize_mrz,
        normalize_td3_line1=normalize_td3_line1,
        normalize_td3_line2=normalize_td3_line2,
        score_td3_line1=score_td3_line1,
        score_td3_line2=score_td3_line2,
    )


def generate_ocr_candidates(
    line_img,
    prefix: str,
    *,
    to_gray,
    prepare_variants,
    ocr_backend: str,
    ocr_configs,
    tesseract_lang: str,
    mrz_whitelist: str,
    paddle_lang: str,
    paddle_use_gpu: bool,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
    score_td3_line1,
    score_td3_line2,
    variants=None,
):
    backends = resolve_ocr_backends(ocr_backend)
    line_kind = "line1" if prefix.startswith("line1_") else "line2" if prefix.startswith("line2_") else None
    if variants is None:
        gray = to_gray(line_img)
        variants = prepare_variants(gray, prefix)

    candidates = []
    seen = set()

    for v in variants:
        if "tesseract" in backends:
            for c in ocr_configs:
                text = ocr_image(
                    v["image"],
                    c["cfg"],
                    tesseract_lang=tesseract_lang,
                    mrz_whitelist=mrz_whitelist,
                    normalize_mrz=normalize_mrz,
                )
                if not text:
                    continue

                norm = normalize_mrz(text)
                key = ("tesseract", norm, v["variant_id"], c["psm"])
                if key in seen:
                    continue
                seen.add(key)

                candidates.append({
                    "backend": "tesseract",
                    "text_raw": text,
                    "text": norm,
                    "variant_id": v["variant_id"],
                    "variant_meta": v["meta"],
                    "psm": c["psm"],
                    "image": v["image"],
                })

    if "paddle" in backends:
        paddle_texts = paddle_ocr_images(
            [v["image"] for v in variants],
            line_kind=line_kind,
            paddle_lang=paddle_lang,
            paddle_use_gpu=paddle_use_gpu,
            normalize_mrz=normalize_mrz,
            normalize_td3_line1=normalize_td3_line1,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
        )
        for v, text in zip(variants, paddle_texts):
            if not text:
                continue
            norm = normalize_mrz(text)
            key = ("paddle", norm, v["variant_id"], PADDLE_OCR_BACKEND_LABEL)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "backend": "paddle",
                "text_raw": text,
                "text": norm,
                "variant_id": v["variant_id"],
                "variant_meta": v["meta"],
                "psm": PADDLE_OCR_BACKEND_LABEL,
                "image": v["image"],
            })

    return candidates


def candidate_support_bonus(support_count: int) -> float:
    if support_count <= 1:
        return 0.0

    return (math.log2(support_count) ** 2) * CANDIDATE_SUPPORT_BONUS_SCALE


def _candidate_support_group(candidate: dict) -> tuple:
    variant_meta = candidate.get("variant_meta") or {}
    return (
        candidate.get("backend"),
        candidate.get("split_label"),
        variant_meta.get("source"),
        variant_meta.get("threshold"),
    )


def apply_candidate_support_bonus(candidates) -> None:
    support_counts = {}
    raw_support_counts = {}
    for cand in candidates:
        text = cand["text"]
        raw_support_counts[text] = raw_support_counts.get(text, 0) + 1
        groups = support_counts.setdefault(text, set())
        groups.add(_candidate_support_group(cand))

    for cand in candidates:
        support_count = len(support_counts[cand["text"]])
        support_bonus = candidate_support_bonus(support_count)
        base_score = cand["score"]

        cand["base_score"] = base_score
        cand["support_count"] = support_count
        cand["raw_support_count"] = raw_support_counts[cand["text"]]
        cand["support_bonus"] = support_bonus
        cand["score"] = base_score + support_bonus


def line1_selection_penalty(candidate: dict) -> float:
    if candidate.get("backend") != "paddle":
        return 0.0

    penalty = 0.0
    if candidate.get("spill_trimmed"):
        penalty += PADDLE_LINE1_SPILL_SELECTION_PENALTY
        if any(
            repair.get("reason") == "name_noise_collapse"
            for repair in candidate.get("repairs", [])
        ):
            penalty += PADDLE_LINE1_SPILL_COLLAPSE_PENALTY

    return penalty
