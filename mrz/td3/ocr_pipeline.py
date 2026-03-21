import os
import re
import math
import sys
import itertools
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from env_utils import load_env_file
from logger_utils import is_debug_enabled
from mrz.td3.country_codes import is_valid_mrz_country_code
from mrz.td3.checksums import (
    AMBIGUOUS_FIELD_SUBS,
    DOC_NUMBER_AMBIGUOUS_SUBS,
    build_checksum_confidence,
    char_value,
    checksum,
    correct_field,
    validate_and_correct_mrz,
    validate_td3_checks,
)
from mrz.td3.normalize import (
    MRZ_LINE_LEN,
    MRZ_WHITELIST,
    _ambiguous_doc_char_count,
    _generate_country_code_variants,
    _normalize_numeric_field,
    _sanitize_alpha,
    _sanitize_name_token,
    _sanitize_name_zone,
    _split_tail_name_tokens,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
)
from mrz.td3.ocr_runner import (
    PADDLE_OCR_BACKEND_LABEL,
    apply_candidate_support_bonus as _apply_candidate_support_bonus,
    best_paddle_text_candidate as _best_paddle_text_candidate_impl,
    extract_ocr_confidence as _extract_ocr_confidence,
    get_paddle_ocr_stats as _get_paddle_ocr_stats_impl,
    generate_ocr_candidates as generate_ocr_candidates_impl,
    get_paddle_ocr as _get_paddle_ocr_impl,
    line1_selection_penalty as _line1_selection_penalty,
    ocr_image as _ocr_image_impl,
    paddle_ocr_image as _paddle_ocr_image_impl,
    paddle_ocr_images as _paddle_ocr_images_impl,
    reset_paddle_ocr_stats as _reset_paddle_ocr_stats_impl,
    resolve_ocr_backends as _resolve_ocr_backends_impl,
    resolve_paddle_use_gpu as _resolve_paddle_use_gpu_impl,
    trim_line1_spill as _trim_line1_spill_impl,
)
from mrz.td3.repair import (
    MIN_LINE1_ZONE_REPAIR_GAIN,
    MIN_TOKEN_REPAIR_SCORE_GAIN,
    TOKEN_AMBIGUOUS_SUBS,
    _build_filler_tail,
    _build_td3_line1,
    _candidate_given_zone_repairs,
    _generate_token_variants,
    _split_given_name_zone,
    build_repair_confidence,
    repair_given_name_token,
    repair_given_name_zone,
    repair_issuing_country_code,
    repair_paddle_line1_candidate as repair_paddle_line1_candidate_impl,
    repair_td3_line1 as repair_td3_line1_impl,
)
from mrz.td3.score import (
    PAIR_COUNTRY_MATCH_BONUS,
    _max_consonant_run,
    _max_vowel_run,
    _name_token_score,
    assemble_td3_confidence,
    build_candidate_margin_confidence,
    build_structure_confidence,
    pair_consistency_bonus,
    score_split_quality,
    score_td3_line1,
    score_td3_line2,
)
from mrz.td3.variants import (
    FAST_OCR_SPLIT_LABELS,
    FAST_OCR_VARIANT_SCALES,
    FAST_OCR_VARIANT_SOURCES,
    FAST_OCR_VARIANT_THRESHOLDS,
    _adaptive_thresh,
    _apply_clahe,
    _cap_width,
    _otsu_thresh,
    _resize,
    _resize_to_height,
    _thicken,
    _to_gray,
    build_split_candidates as build_split_candidates_impl,
    clean_mrz_image,
    estimate_ocr_search_space as estimate_ocr_search_space_impl,
    ocr_search_profile,
    prepare_variants,
    split_mrz_lines as split_mrz_lines_impl,
    split_mrz_lines_at as split_mrz_lines_at_impl,
)
from ocr_backends.tesseract_backend import (
    build_ocr_configs,
    configure_tesseract_cmd,
    resolve_oems as resolve_tesseract_oems,
)
from ocr_backends.paddle_backend import extract_paddle_lines as paddle_backend_extract_lines


load_env_file()

warnings.filterwarnings(
    "ignore",
    message=r".*RequestsDependencyWarning: urllib3.*|urllib3 .* doesn't match a supported version.*",
    module=r"requests(\..*)?",
)

warnings.filterwarnings(
    "ignore",
    message=r".*No ccache found.*",
    category=UserWarning,
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

resolved_tesseract_cmd = configure_tesseract_cmd()

TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng").strip() or "eng"
OCR_BACKEND = os.getenv("OCR_BACKEND", "paddle").strip().lower() or "paddle"
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en").strip() or "en"


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Ignoring invalid integer in {name}: {raw}")
        return default

    return value if value > 0 else default


def _parse_choice_env(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in allowed:
        return raw
    print(f"[WARN] Ignoring invalid value in {name}: {raw}")
    return default


PADDLEOCR_USE_GPU = _parse_bool_env("PADDLEOCR_USE_GPU", False)


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default[:]

    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            print(f"[WARN] Ignoring invalid integer in {name}: {token}")

    return values or default[:]


def _resolve_oems(lang: str) -> list[int]:
    requested = _parse_int_list_env("TESSERACT_OEMS", [1])
    return resolve_tesseract_oems(lang, requested)


FAST_OCR = _parse_bool_env("FAST_OCR", False)
PADDLE_FAST = _parse_bool_env("PADDLE_FAST", True)
PADDLE_PROFILE = _parse_choice_env("PADDLE_PROFILE", "exhaustive", {"exhaustive", "balanced", "fast"})
MRZ_VARIANT_WORKERS = _parse_positive_int_env("MRZ_VARIANT_WORKERS", min(4, os.cpu_count() or 1))
TESSERACT_PSMS = _parse_int_list_env(
    "TESSERACT_PSMS",
    [7] if FAST_OCR else [7, 6, 13],
)
TESSERACT_OEMS = _resolve_oems(TESSERACT_LANG)

OCR_CONFIGS = build_ocr_configs(TESSERACT_OEMS, TESSERACT_PSMS)
ESSENTIAL_OUTPUTS = {
    "mrz_clean.png",
    "mrz_line1.png",
    "mrz_line2.png",
    "best_variant_line1.png",
    "best_variant_line2.png",
}


# -------------------------------------------------------
# BASIC IO
# -------------------------------------------------------

def save(img, filename):
    if not is_debug_enabled() and filename not in ESSENTIAL_OUTPUTS:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[Save]  {path}")


# -------------------------------------------------------
# NORMALIZATION / CHECKSUMS
# -------------------------------------------------------


# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------

def _ocr_search_profile() -> dict:
    return ocr_search_profile(
        FAST_OCR,
        paddle_fast=(PADDLE_FAST or PADDLE_PROFILE == "fast"),
        paddle_profile=PADDLE_PROFILE,
        ocr_backend=OCR_BACKEND,
    )


def _prepare_variants(gray, prefix: str, profile: dict | None = None):
    active_profile = dict(profile or _ocr_search_profile())
    active_profile.setdefault("variant_workers", MRZ_VARIANT_WORKERS)
    return prepare_variants(gray, prefix, active_profile)


# -------------------------------------------------------
# LINE SPLIT
# -------------------------------------------------------

def split_mrz_lines(img):
    return split_mrz_lines_impl(img, save_debug=save)


def split_mrz_lines_at(gray, split_y: int):
    return split_mrz_lines_at_impl(gray, split_y)


def build_split_candidates(gray, base_meta: dict, profile: dict | None = None):
    return build_split_candidates_impl(gray, base_meta, profile or _ocr_search_profile())


def estimate_ocr_search_space() -> dict:
    return estimate_ocr_search_space_impl(
        fast_ocr=FAST_OCR,
        tesseract_psms=TESSERACT_PSMS,
        tesseract_oems=TESSERACT_OEMS,
        paddle_fast=(PADDLE_FAST or PADDLE_PROFILE == "fast"),
        paddle_profile=PADDLE_PROFILE,
        ocr_backend=OCR_BACKEND,
    )


def _should_early_accept_paddle_split(split_info: dict, line1_ranked: list[dict], line2_ranked: list[dict]) -> bool:
    if OCR_BACKEND != "paddle" or not PADDLE_FAST:
        return False
    if split_info.get("label") != "projection":
        return False
    if not line1_ranked or not line2_ranked:
        return False

    best_line1 = line1_ranked[0]
    best_line2 = line2_ranked[0]
    checks = best_line2.get("checks", {})

    return (
        checks.get("passed_count", 0) == 5
        and checks.get("composite_valid", False)
        and best_line1.get("base_score", best_line1.get("score", 0.0)) >= 105.0
        and best_line2.get("score", 0.0) >= 150.0
    )


def _auto_requires_tesseract(line1_candidates: list[dict], line2_candidates: list[dict]) -> bool:
    if OCR_BACKEND != "auto":
        return False
    if not line1_candidates or not line2_candidates:
        return True

    line1_ranked = _rank_candidates(list(line1_candidates), "score")
    line2_ranked = _rank_candidates(list(line2_candidates), "score")
    best_line1 = line1_ranked[0]
    best_line2 = line2_ranked[0]
    checks = best_line2.get("checks", {})

    return not (
        best_line1.get("score", 0.0) >= 105.0
        and best_line2.get("score", 0.0) >= 150.0
        and checks.get("passed_count", 0) == 5
        and checks.get("composite_valid", False)
    )


def _run_auto_paddle_batches(prepared_splits: list[dict]) -> list[dict]:
    split_results = [{"line1": [], "line2": []} for _ in prepared_splits]
    line1_jobs: list[tuple[int, dict]] = []
    line1_images: list[np.ndarray] = []
    line2_jobs: list[tuple[int, dict]] = []
    line2_images: list[np.ndarray] = []

    for split_index, prepared in enumerate(prepared_splits):
        for variant in prepared["line1_variants"]:
            line1_jobs.append((split_index, variant))
            line1_images.append(variant["image"])
        for variant in prepared["line2_variants"]:
            line2_jobs.append((split_index, variant))
            line2_images.append(variant["image"])

    if line1_images:
        line1_texts = _paddle_ocr_images_impl(
            line1_images,
            line_kind="line1",
            paddle_lang=PADDLEOCR_LANG,
            paddle_use_gpu=PADDLEOCR_USE_GPU,
            normalize_mrz=normalize_mrz,
            normalize_td3_line1=normalize_td3_line1,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
        )
        for (split_index, variant), text in zip(line1_jobs, line1_texts):
            if not text:
                continue
            split_results[split_index]["line1"].append({
                "backend": "paddle",
                "text_raw": text,
                "text": normalize_mrz(text),
                "variant_id": variant["variant_id"],
                "variant_meta": variant["meta"],
                "psm": PADDLE_OCR_BACKEND_LABEL,
                "image": variant["image"],
            })

    if line2_images:
        line2_texts = _paddle_ocr_images_impl(
            line2_images,
            line_kind="line2",
            paddle_lang=PADDLEOCR_LANG,
            paddle_use_gpu=PADDLEOCR_USE_GPU,
            normalize_mrz=normalize_mrz,
            normalize_td3_line1=normalize_td3_line1,
            normalize_td3_line2=normalize_td3_line2,
            score_td3_line1=score_td3_line1,
            score_td3_line2=score_td3_line2,
        )
        for (split_index, variant), text in zip(line2_jobs, line2_texts):
            if not text:
                continue
            split_results[split_index]["line2"].append({
                "backend": "paddle",
                "text_raw": text,
                "text": normalize_mrz(text),
                "variant_id": variant["variant_id"],
                "variant_meta": variant["meta"],
                "psm": PADDLE_OCR_BACKEND_LABEL,
                "image": variant["image"],
            })

    return split_results


# -------------------------------------------------------
# OCR
# -------------------------------------------------------

def _ocr_image(img, cfg: str) -> str:
    return _ocr_image_impl(
        img,
        cfg,
        tesseract_lang=TESSERACT_LANG,
        mrz_whitelist=MRZ_WHITELIST,
        normalize_mrz=normalize_mrz,
    )


def _resolve_ocr_backends() -> list[str]:
    return _resolve_ocr_backends_impl(OCR_BACKEND)


def _trim_line1_spill(text: str) -> str:
    return _trim_line1_spill_impl(text, normalize_td3_line1)


def _resolve_paddle_use_gpu() -> bool:
    return _resolve_paddle_use_gpu_impl(PADDLEOCR_USE_GPU)


def _get_paddle_ocr():
    return _get_paddle_ocr_impl(PADDLEOCR_LANG, PADDLEOCR_USE_GPU)


def _reset_paddle_ocr_stats() -> None:
    _reset_paddle_ocr_stats_impl()


def _get_paddle_ocr_stats() -> dict:
    return _get_paddle_ocr_stats_impl()


def _best_paddle_text_candidate(texts: list[str], line_kind: str) -> str:
    return _best_paddle_text_candidate_impl(
        texts,
        line_kind,
        normalize_mrz=normalize_mrz,
        normalize_td3_line2=normalize_td3_line2,
        score_td3_line1=score_td3_line1,
        score_td3_line2=score_td3_line2,
        trim_line1_spill_func=_trim_line1_spill,
    )


def _paddle_ocr_image(img, line_kind: str | None = None) -> str:
    ocr = _get_paddle_ocr()
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
        return _best_paddle_text_candidate(texts, line_kind)

    return normalize_mrz("".join(texts))


def generate_ocr_candidates(line_img, prefix: str):
    return generate_ocr_candidates_impl(
        line_img,
        prefix,
        to_gray=_to_gray,
        prepare_variants=_prepare_variants,
        ocr_backend=OCR_BACKEND,
        ocr_configs=OCR_CONFIGS,
        tesseract_lang=TESSERACT_LANG,
        mrz_whitelist=MRZ_WHITELIST,
        paddle_lang=PADDLEOCR_LANG,
        paddle_use_gpu=PADDLEOCR_USE_GPU,
        normalize_mrz=normalize_mrz,
        normalize_td3_line1=normalize_td3_line1,
        normalize_td3_line2=normalize_td3_line2,
        score_td3_line1=score_td3_line1,
        score_td3_line2=score_td3_line2,
        variants=None,
    )


# -------------------------------------------------------
# SCORING / REPAIR
# -------------------------------------------------------

def repair_td3_line1(line1: str) -> tuple[str, list[dict]]:
    return repair_td3_line1_impl(line1, trim_line1_spill_func=_trim_line1_spill)


def _repair_paddle_line1_candidate(line1: str) -> tuple[str, dict | None]:
    return repair_paddle_line1_candidate_impl(line1, trim_line1_spill_func=_trim_line1_spill)


# -------------------------------------------------------
# PARSING
# -------------------------------------------------------

def parse_mrz_fields(line1: str, line2: str) -> dict:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    parsed = {}

    if len(line1) >= 5:
        parsed["document_type"] = line1[0]
        parsed["issuing_country"] = line1[2:5].replace("<", "")

        name_zone = line1[5:]
        if "<<" in name_zone:
            surname, given = name_zone.split("<<", 1)
            parsed["surname"] = surname.replace("<", "")
            parsed["given_names"] = given.replace("<", " ").strip().replace("  ", " ")
        else:
            parsed["surname"] = ""
            parsed["given_names"] = ""

    if len(line2) >= MRZ_LINE_LEN:
        parsed["document_number"] = line2[0:9].replace("<", "")
        parsed["document_number_check"] = line2[9]
        parsed["nationality"] = line2[10:13].replace("<", "")
        parsed["birth_date_yymmdd"] = line2[13:19]
        parsed["birth_date_check"] = line2[19]
        parsed["sex"] = line2[20]
        parsed["expiry_date_yymmdd"] = line2[21:27]
        parsed["expiry_date_check"] = line2[27]
        parsed["personal_number"] = line2[28:42].replace("<", "")
        parsed["personal_number_check"] = line2[42]
        parsed["final_check"] = line2[43]

    return parsed


# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

def _pick_top(candidates, score_key: str, topn: int = 5):
    ordered = sorted(candidates, key=lambda x: x[score_key], reverse=True)
    return ordered[:topn]


def _rank_candidates(candidates, score_key: str):
    ordered = sorted(candidates, key=lambda x: x[score_key], reverse=True)
    for rank, cand in enumerate(ordered):
        cand["candidate_rank"] = rank
    return ordered


def _pair_selection_key(candidate_pair: dict) -> tuple:
    line1_rank = candidate_pair["line1"].get("candidate_rank", 10**9)
    line2_rank = candidate_pair["line2"].get("candidate_rank", 10**9)

    return (
        candidate_pair["pair_score"],
        candidate_pair.get("pair_bonus", 0.0),
        candidate_pair["line1"]["score"],
        candidate_pair["line2"]["score"],
        -line1_rank,
        -line2_rank,
        candidate_pair["line1"]["text"],
        candidate_pair["line2"]["text"],
    )


def _serialize_line1_candidate(c: dict) -> dict:
    return {
        "text": c["text"],
        "score": round(c["score"], 2),
        "base_score": round(c.get("base_score", c["score"]), 2),
        "selection_penalty": round(c.get("selection_penalty", 0.0), 2),
        "variant_id": c["variant_id"],
        "psm": c["psm"],
        "checksum_pass_count": c["checksum_pass_count"],
        "support_count": c.get("support_count", 1),
        "raw_support_count": c.get("raw_support_count", c.get("support_count", 1)),
        "support_bonus": round(c.get("support_bonus", 0.0), 2),
        "split_label": c.get("split_label"),
        "split_y": c.get("split_y"),
        "variant_meta": c.get("variant_meta"),
        "backend": c.get("backend"),
        "spill_trimmed": bool(c.get("spill_trimmed", False)),
        "repairs": list(c.get("repairs", [])),
    }


def _serialize_line2_candidate(c: dict) -> dict:
    return {
        "text": c["text"],
        "score": round(c["score"], 2),
        "variant_id": c["variant_id"],
        "psm": c["psm"],
        "checksum_pass_count": c["checksum_pass_count"],
        "split_label": c.get("split_label"),
        "split_y": c.get("split_y"),
        "variant_meta": c.get("variant_meta"),
        "backend": c.get("backend"),
        "checks": c.get("checks"),
    }


def run_ocr(mrz_img):
    ocr_started_at = time.perf_counter()
    _reset_paddle_ocr_stats()
    if isinstance(mrz_img, str):
        img = cv2.imread(mrz_img, cv2.IMREAD_COLOR)
    else:
        img = mrz_img

    if img is None:
        raise RuntimeError("Cannot load MRZ image")

    gray = _to_gray(img)
    save(gray, "mrz_gray.png")

    clahe = _apply_clahe(gray)
    save(clahe, "mrz_clahe.png")

    th_otsu = _otsu_thresh(_resize(gray, 3))
    save(th_otsu, "mrz_thresh_otsu.png")

    th_adaptive = _adaptive_thresh(_resize(gray, 3))
    save(th_adaptive, "mrz_thresh_adaptive.png")

    line1_img_base, line2_img_base, split_meta = split_mrz_lines(gray)

    search_space = estimate_ocr_search_space()
    print(
        "[OCR] Search profile: "
        f"fast={search_space['fast_ocr']} "
        f"paddle_fast={search_space.get('paddle_fast', False)} "
        f"profile={search_space.get('profile_name')} "
        f"psms={search_space['psms']} "
        f"splits={search_space['split_count']} "
        f"per_line_variants={search_space['per_line_variants']} "
        f"estimated_tesseract_calls={search_space['total_tesseract_calls']}"
    )

    split_candidates = build_split_candidates(gray, split_meta)
    all_line1_candidates = []
    all_line2_candidates = []
    split_summaries = []
    auto_stats = {
        "paddle_only_splits": 0,
        "tesseract_fallback_splits": 0,
    }
    candidate_generation_started_at = time.perf_counter()
    variant_preparation_started_at = time.perf_counter()

    def prepare_split_payload(split_info: dict) -> dict:
        split_y = split_info["split_y"]
        line1_img, line2_img = split_mrz_lines_at(gray, split_y)
        return {
            "split_info": split_info,
            "line1_img": line1_img,
            "line2_img": line2_img,
            "line1_variants": _prepare_variants(line1_img, f"line1_{split_info['label']}"),
            "line2_variants": _prepare_variants(line2_img, f"line2_{split_info['label']}"),
        }

    prepared_splits = []
    if len(split_candidates) <= 1:
        prepared_splits = [prepare_split_payload(split_candidates[0])] if split_candidates else []
    else:
        max_workers = min(MRZ_VARIANT_WORKERS, len(split_candidates), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            prepared_splits = list(executor.map(prepare_split_payload, split_candidates))

    variant_preparation_ms = round((time.perf_counter() - variant_preparation_started_at) * 1000.0, 2)
    ocr_candidate_eval_started_at = time.perf_counter()
    auto_paddle_candidates = _run_auto_paddle_batches(prepared_splits) if OCR_BACKEND == "auto" else None

    for split_index, prepared in enumerate(prepared_splits):
        split_info = prepared["split_info"]
        line1_img = prepared["line1_img"]
        line2_img = prepared["line2_img"]
        if OCR_BACKEND == "auto":
            line1_candidates = list(auto_paddle_candidates[split_index]["line1"])
            line2_candidates = list(auto_paddle_candidates[split_index]["line2"])
        else:
            line1_candidates = generate_ocr_candidates_impl(
                line1_img,
                f"line1_{split_info['label']}",
                to_gray=_to_gray,
                prepare_variants=_prepare_variants,
                ocr_backend=OCR_BACKEND,
                ocr_configs=OCR_CONFIGS,
                tesseract_lang=TESSERACT_LANG,
                mrz_whitelist=MRZ_WHITELIST,
                paddle_lang=PADDLEOCR_LANG,
                paddle_use_gpu=PADDLEOCR_USE_GPU,
                normalize_mrz=normalize_mrz,
                normalize_td3_line1=normalize_td3_line1,
                normalize_td3_line2=normalize_td3_line2,
                score_td3_line1=score_td3_line1,
                score_td3_line2=score_td3_line2,
                variants=prepared["line1_variants"],
            )
            line2_candidates = generate_ocr_candidates_impl(
                line2_img,
                f"line2_{split_info['label']}",
                to_gray=_to_gray,
                prepare_variants=_prepare_variants,
                ocr_backend=OCR_BACKEND,
                ocr_configs=OCR_CONFIGS,
                tesseract_lang=TESSERACT_LANG,
                mrz_whitelist=MRZ_WHITELIST,
                paddle_lang=PADDLEOCR_LANG,
                paddle_use_gpu=PADDLEOCR_USE_GPU,
                normalize_mrz=normalize_mrz,
                normalize_td3_line1=normalize_td3_line1,
                normalize_td3_line2=normalize_td3_line2,
                score_td3_line1=score_td3_line1,
                score_td3_line2=score_td3_line2,
                variants=prepared["line2_variants"],
            )

        def _prepare_line1_candidates(candidates: list[dict]) -> list[dict]:
            prepared_candidates = []
            for cand in candidates:
                normalized_raw = normalize_td3_line1(cand["text"])
                text = _trim_line1_spill(cand["text"])
                spill_trimmed = text != normalized_raw
                repaired, repairs = repair_td3_line1(text)
                if cand.get("backend") == "paddle":
                    paddle_repaired, paddle_meta = _repair_paddle_line1_candidate(repaired)
                    if paddle_meta is not None:
                        repaired = paddle_repaired
                        repairs = repairs + [paddle_meta]
                cand["text"] = _trim_line1_spill(repaired)
                cand["repairs"] = repairs
                cand["spill_trimmed"] = spill_trimmed
                cand["selection_penalty"] = 0.0
                base_score = score_td3_line1(cand["text"])
                selection_penalty = _line1_selection_penalty(cand)
                cand["selection_penalty"] = selection_penalty
                cand["score"] = base_score - selection_penalty
                cand["checksum_pass_count"] = None
                cand["split_label"] = split_info["label"]
                cand["split_y"] = split_info["split_y"]
                prepared_candidates.append(cand)
            return prepared_candidates

        def _prepare_line2_candidates(candidates: list[dict]) -> list[dict]:
            prepared_candidates = []
            for cand in candidates:
                text = normalize_td3_line2(cand["text"])
                _, repaired, checks = validate_and_correct_mrz("", text)
                cand["text"] = repaired
                cand["checks"] = checks
                cand["score"] = score_td3_line2(repaired)[0]
                cand["checksum_pass_count"] = checks["passed_count"]
                cand["split_label"] = split_info["label"]
                cand["split_y"] = split_info["split_y"]
                prepared_candidates.append(cand)
            return prepared_candidates

        line1_candidates = _prepare_line1_candidates(line1_candidates)
        line2_candidates = _prepare_line2_candidates(line2_candidates)

        if OCR_BACKEND == "auto":
            if _auto_requires_tesseract(line1_candidates, line2_candidates):
                auto_stats["tesseract_fallback_splits"] += 1
                line1_candidates.extend(
                    _prepare_line1_candidates(
                        generate_ocr_candidates_impl(
                            line1_img,
                            f"line1_{split_info['label']}",
                            to_gray=_to_gray,
                            prepare_variants=_prepare_variants,
                            ocr_backend="tesseract",
                            ocr_configs=OCR_CONFIGS,
                            tesseract_lang=TESSERACT_LANG,
                            mrz_whitelist=MRZ_WHITELIST,
                            paddle_lang=PADDLEOCR_LANG,
                            paddle_use_gpu=PADDLEOCR_USE_GPU,
                            normalize_mrz=normalize_mrz,
                            normalize_td3_line1=normalize_td3_line1,
                            normalize_td3_line2=normalize_td3_line2,
                            score_td3_line1=score_td3_line1,
                            score_td3_line2=score_td3_line2,
                            variants=prepared["line1_variants"],
                        )
                    )
                )
                line2_candidates.extend(
                    _prepare_line2_candidates(
                        generate_ocr_candidates_impl(
                            line2_img,
                            f"line2_{split_info['label']}",
                            to_gray=_to_gray,
                            prepare_variants=_prepare_variants,
                            ocr_backend="tesseract",
                            ocr_configs=OCR_CONFIGS,
                            tesseract_lang=TESSERACT_LANG,
                            mrz_whitelist=MRZ_WHITELIST,
                            paddle_lang=PADDLEOCR_LANG,
                            paddle_use_gpu=PADDLEOCR_USE_GPU,
                            normalize_mrz=normalize_mrz,
                            normalize_td3_line1=normalize_td3_line1,
                            normalize_td3_line2=normalize_td3_line2,
                            score_td3_line1=score_td3_line1,
                            score_td3_line2=score_td3_line2,
                            variants=prepared["line2_variants"],
                        )
                    )
                )
            else:
                auto_stats["paddle_only_splits"] += 1

        for cand in line1_candidates:
            all_line1_candidates.append(cand)

        for cand in line2_candidates:
            all_line2_candidates.append(cand)

        if not line1_candidates or not line2_candidates:
            continue

        line1_ranked = _rank_candidates(line1_candidates, "score")
        line2_ranked = _rank_candidates(line2_candidates, "score")
        best_line2_for_split = line2_ranked[0]
        split_score = score_split_quality(
            split_info,
            best_line2_for_split["checks"],
            best_line2_for_split["score"],
        )
        split_summaries.append({
            "split_info": split_info,
            "split_score": split_score,
            "line1_candidate_count": len(line1_candidates),
            "line2_candidate_count": len(line2_candidates),
            "line1_img": line1_img,
            "line2_img": line2_img,
        })

        if _should_early_accept_paddle_split(split_info, line1_ranked, line2_ranked):
            break

    ocr_candidate_eval_ms = round((time.perf_counter() - ocr_candidate_eval_started_at) * 1000.0, 2)
    candidate_generation_ms = round((time.perf_counter() - candidate_generation_started_at) * 1000.0, 2)

    if not all_line1_candidates or not all_line2_candidates:
        raise RuntimeError("No valid split candidates produced OCR results")

    # ============================
    # RANKING
    # ============================

    ranking_started_at = time.perf_counter()

    _apply_candidate_support_bonus(all_line1_candidates)

    line1_ranked = _rank_candidates(all_line1_candidates, "score")
    line2_ranked = _rank_candidates(all_line2_candidates, "score")

    line1_top = line1_ranked[:5]
    line2_top = line2_ranked[:5]

    # ============================
    # VALIDATION-GATED PAIR SELECTION
    # ============================

    valid_pairs = []
    fallback_pairs = []
    pair_count = 0

    for c1 in line1_ranked:
        for c2 in line2_ranked:
            pair_count += 1

            checks = c2.get("checks") or validate_td3_checks(c2["text"])
            pair_bonus = pair_consistency_bonus(c1["text"], c2["text"])

            pair_score = (
                c1["score"]
                + c2["score"]
                + (checks["passed_count"] * 4.0)
                + pair_bonus
            )

            candidate_pair = {
                "line1": c1,
                "line2": c2,
                "pair_score": pair_score,
                "pair_bonus": pair_bonus,
                "checks": checks,
                "repairs_applied": list(c1.get("repairs", [])),
            }

            if checks.get("passed_count") == 5 and checks.get("composite_valid"):
                valid_pairs.append(candidate_pair)
            else:
                fallback_pairs.append(candidate_pair)

    def _select_best(pairs):
        best = None
        for p in pairs:
            if best is None or _pair_selection_key(p) > _pair_selection_key(best):
                best = p
        return best

    # ============================
    # SELECTION STRATEGY
    # ============================

    if valid_pairs:
        best_pair = _select_best(valid_pairs)
        best_pair["selection_mode"] = "strict_valid"
    else:
        best_pair = _select_best(fallback_pairs)
        best_pair["selection_mode"] = "fallback"
        best_pair["force_suspicious"] = True


    if best_pair is None:
        raise RuntimeError("No valid line pair selected from OCR candidates")

    ranking_and_pairing_ms = round(
        (time.perf_counter() - ranking_started_at) * 1000.0, 2
    )

    selected_line1_split = best_pair["line1"]["split_label"]
    selected_line2_split = best_pair["line2"]["split_label"]

    selected_line1_y = best_pair["line1"]["split_y"]
    selected_line2_y = best_pair["line2"]["split_y"]

    selected_line1_img, _ = split_mrz_lines_at(gray, selected_line1_y)
    _, selected_line2_img = split_mrz_lines_at(gray, selected_line2_y)

    selected_split_summary = next(
        (
            summary
            for summary in split_summaries
            if summary["split_info"]["label"] == selected_line2_split
        ),
        None,
    )

    best_line1 = best_pair["line1"]["text"]
    best_line2 = best_pair["line2"]["text"]
    checks = best_pair["checks"]

    checksum_confidence = build_checksum_confidence(checks)
    structure_confidence = build_structure_confidence(best_line1, best_line2)
    repair_confidence = build_repair_confidence(best_pair["repairs_applied"])
    margin_confidence = build_candidate_margin_confidence(
        best_pair, line1_ranked, line2_ranked
    )

    ocr_confidence = (
        _extract_ocr_confidence(best_pair["line2"])
        or _extract_ocr_confidence(best_pair["line1"])
    )

    confidence = assemble_td3_confidence(
        checksum_confidence=checksum_confidence,
        structure_confidence=structure_confidence,
        repair_confidence=repair_confidence,
        margin_confidence=margin_confidence,
        ocr_confidence=ocr_confidence,
        selected_meta={
            "line1_score": best_pair["line1"]["score"],
            "line2_score": best_pair["line2"]["score"],
            "pair_score": best_pair["pair_score"],
        },
    )

    save(selected_line1_img, "mrz_line1.png")
    save(selected_line2_img, "mrz_line2.png")
    save(best_pair["line1"]["image"], "best_variant_line1.png")
    save(best_pair["line2"]["image"], "best_variant_line2.png")

    parsed = parse_mrz_fields(best_line1, best_line2)

    ocr_meta = {
        "split": {
            **split_meta,
            "selected_label": selected_line2_split,
            "selected_split_y": selected_line2_y,
            "selected_top_ratio": round(selected_split_summary["split_info"]["top_ratio"], 3)
            if selected_split_summary is not None else round(selected_line2_y / gray.shape[0], 3),
            "selected_bottom_ratio": round(selected_split_summary["split_info"]["bottom_ratio"], 3)
            if selected_split_summary is not None else round((gray.shape[0] - selected_line2_y) / gray.shape[0], 3),
            "selected_split_score": round(selected_split_summary["split_score"], 2)
            if selected_split_summary is not None else None,
            "selected_line1_label": selected_line1_split,
            "selected_line1_split_y": selected_line1_y,
            "selected_line2_label": selected_line2_split,
            "selected_line2_split_y": selected_line2_y,
        },
        "candidate_stats": {
            "line1_candidates": len(line1_ranked),
            "line2_candidates": len(line2_ranked),
            "pairs_evaluated": pair_count,
            "line1_top": [_serialize_line1_candidate(c) for c in line1_top],
            "line2_top": [_serialize_line2_candidate(c) for c in line2_top],
            "line1_all": [_serialize_line1_candidate(c) for c in line1_ranked],
            "line2_all": [_serialize_line2_candidate(c) for c in line2_ranked],
        },
        "selected": {
            "line1": {
                "score": round(best_pair["line1"]["score"], 2),
                "base_score": round(
                    best_pair["line1"].get("base_score", best_pair["line1"]["score"]),
                    2,
                ),
                "selection_penalty": round(
                    best_pair["line1"].get("selection_penalty", 0.0),
                    2,
                ),
                "text": best_line1,
                "variant_id": best_pair["line1"]["variant_id"],
                "psm": best_pair["line1"]["psm"],
                "split_label": selected_line1_split,
                "variant_meta": best_pair["line1"]["variant_meta"],
                "support_count": best_pair["line1"].get("support_count", 1),
                "raw_support_count": best_pair["line1"].get(
                    "raw_support_count",
                    best_pair["line1"].get("support_count", 1),
                ),
                "support_bonus": round(best_pair["line1"].get("support_bonus", 0.0), 2),
            },
            "line2": {
                "score": round(best_pair["line2"]["score"], 2),
                "text": best_line2,
                "variant_id": best_pair["line2"]["variant_id"],
                "psm": best_pair["line2"]["psm"],
                "split_label": selected_line2_split,
                "variant_meta": best_pair["line2"]["variant_meta"],
            },
            "pair_score": round(best_pair["pair_score"], 2),
            "pair_bonus": round(best_pair.get("pair_bonus", 0.0), 2),
        },
        "checksum_summary": checks,
        "confidence": confidence,
        "repairs_applied": best_pair["repairs_applied"],
        "parsed_fields": parsed,
        "backend_stats": {
            "backend": OCR_BACKEND,
            "paddle": _get_paddle_ocr_stats() if OCR_BACKEND in {"paddle", "auto", "hybrid", "both"} else None,
            "auto": auto_stats if OCR_BACKEND == "auto" else None,
        },
        "timing_ms": {
            "total_ocr_ms": round((time.perf_counter() - ocr_started_at) * 1000.0, 2),
            "candidate_generation_ms": candidate_generation_ms,
            "variant_preparation_ms": variant_preparation_ms,
            "candidate_ocr_ms": ocr_candidate_eval_ms,
            "ranking_and_pairing_ms": ranking_and_pairing_ms,
        },
    }

    print("\n[OCR] Candidate winner")
    print("------------------------------------------------------------")
    print(f"line1: {best_line1}")
    print(f"line2: {best_line2}")
    print(f"pair score: {ocr_meta['selected']['pair_score']:.2f}")
    print(
        f"line1 variant: {ocr_meta['selected']['line1']['variant_id']}  "
        f"psm={ocr_meta['selected']['line1']['psm']}"
    )
    print(
        f"line2 variant: {ocr_meta['selected']['line2']['variant_id']}  "
        f"psm={ocr_meta['selected']['line2']['psm']}"
    )
    print(
        f"checks passed: {checks['passed_count']}/{checks['total_count']} "
        f"(composite={checks['composite_valid']})"
    )
    print(f"repairs applied: {len(ocr_meta['repairs_applied'])}")
    for repair in ocr_meta["repairs_applied"]:
        print(f"  - {repair}")
    print("------------------------------------------------------------")

    print("\nFinal MRZ")
    print("--------------------------------------------")
    print(best_line1)
    print(best_line2)
    print("--------------------------------------------")

    return best_line1, best_line2, ocr_meta


# -------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    mrz_image = os.path.join(OUTPUT_DIR, "mrz_region.png")
    run_ocr(mrz_image)
