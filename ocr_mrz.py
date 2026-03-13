import os
import re
import math
import sys
import shutil
import itertools
import inspect
import io
import contextlib
import warnings
import cv2
import pytesseract
import numpy as np

from env_utils import load_env_file
from mrz_country_codes import is_valid_mrz_country_code


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

# -------------------------------------------------------
# TESSERACT RESOLUTION
# -------------------------------------------------------

WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

def _resolve_tesseract_cmd() -> str | None:
    env_tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    candidates = []

    if env_tesseract_cmd:
        candidates.append(env_tesseract_cmd)

    candidates.append("tesseract")

    if os.name == "nt":
        candidates.extend(WINDOWS_TESSERACT_PATHS)

    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        resolved = shutil.which(expanded)
        if resolved:
            return resolved

    if env_tesseract_cmd:
        print(f"[WARN] TESSERACT_CMD not found: {env_tesseract_cmd}")

    return None


resolved_tesseract_cmd = _resolve_tesseract_cmd()
if resolved_tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract_cmd


MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_LINE_LEN = 44
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng").strip() or "eng"
OCR_BACKEND = os.getenv("OCR_BACKEND", "tesseract").strip().lower() or "tesseract"
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en").strip() or "en"


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


PADDLEOCR_USE_GPU = _parse_bool_env("PADDLEOCR_USE_GPU", False)


def _resolve_paddle_cache_home() -> str:
    env_cache = os.getenv("PADDLE_PDX_CACHE_HOME", "").strip()
    if env_cache:
        return env_cache

    return os.path.abspath(os.path.join(os.getcwd(), ".paddlex"))


def _ensure_stub_ccache_on_path() -> None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        if path and os.path.exists(os.path.join(path, "ccache")):
            return

    stub_dir = os.path.join(_resolve_paddle_cache_home(), "bin")
    os.makedirs(stub_dir, exist_ok=True)
    stub_path = os.path.join(stub_dir, "ccache")

    if not os.path.exists(stub_path):
        with open(stub_path, "w", encoding="ascii") as f:
            f.write("#!/bin/sh\n")
            f.write('exec "$@"\n')
        os.chmod(stub_path, 0o755)

    os.environ["PATH"] = stub_dir + os.pathsep + os.environ.get("PATH", "")


def _should_disable_paddle_model_source_check(paddle_cache_home: str) -> bool:
    env_value = os.getenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "").strip()
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}

    official_models_dir = os.path.join(paddle_cache_home, "official_models")
    return os.path.isdir(official_models_dir)


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


def _legacy_oem_supported(lang: str) -> bool:
    probe = np.full((24, 96), 255, dtype=np.uint8)

    try:
        pytesseract.image_to_string(
            probe,
            lang=lang,
            config=(
                "--oem 0 --psm 10 "
                "-c tessedit_char_whitelist=A "
                "-c load_system_dawg=0 "
                "-c load_freq_dawg=0"
            ),
        )
        return True
    except pytesseract.TesseractError as exc:
        message = str(exc).lower()
        if "legacy" in message and "components are not present" in message:
            return False
        print(f"[WARN] Could not validate legacy OCR support for '{lang}': {exc}")
        return False


def _resolve_oems(lang: str) -> list[int]:
    requested = _parse_int_list_env("TESSERACT_OEMS", [1])
    resolved = []
    legacy_checked = False
    legacy_supported = False

    for oem in requested:
        if oem not in (0, 1, 2, 3):
            print(f"[WARN] Ignoring unsupported TESSERACT_OEMS value: {oem}")
            continue

        if oem in (0, 2):
            if not legacy_checked:
                legacy_supported = _legacy_oem_supported(lang)
                legacy_checked = True
            if not legacy_supported:
                print(
                    f"[WARN] Skipping OEM {oem} for '{lang}' because legacy components are not available"
                )
                continue

        if oem not in resolved:
            resolved.append(oem)

    if resolved:
        return resolved

    print("[WARN] No usable Tesseract OEM configured; falling back to OEM 1")
    return [1]


FAST_OCR = _parse_bool_env("FAST_OCR", False)
TESSERACT_PSMS = _parse_int_list_env(
    "TESSERACT_PSMS",
    [7] if FAST_OCR else [7, 6, 13],
)
TESSERACT_OEMS = _resolve_oems(TESSERACT_LANG)

OCR_CONFIGS = [
    {"oem": oem, "psm": psm, "cfg": f"--oem {oem} --psm {psm}"}
    for oem in TESSERACT_OEMS
    for psm in TESSERACT_PSMS
]

AMBIGUOUS_FIELD_SUBS = {
    "O": ("0",),
    "0": ("O", "Q"),
    "Q": ("0",),
    "I": ("1", "L"),
    "1": ("I", "L"),
    "L": ("I", "1"),
    "Z": ("2",),
    "2": ("Z",),
    "S": ("5",),
    "5": ("S",),
    "B": ("8",),
    "8": ("B",),
    "G": ("6",),
    "6": ("G",),
}

TOKEN_AMBIGUOUS_SUBS = {
    "T": ("I",),
    "I": ("L",),
    "L": ("I",),
    "M": ("N",),
    "N": ("M",),
}

COUNTRY_CODE_AMBIGUOUS_SUBS = {
    "0": ("O", "Q"),
    "1": ("I",),
    "2": ("Z",),
    "5": ("S",),
    "6": ("G",),
    "8": ("B",),
}

NUMERIC_FIELD_SUBS = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

AMBIGUOUS_DOC_CHARS = set(NUMERIC_FIELD_SUBS)
DOC_NUMBER_AMBIGUOUS_SUBS = {
    **AMBIGUOUS_FIELD_SUBS,
    "Q": ("0", "O"),
}

MIN_TOKEN_REPAIR_SCORE_GAIN = 6.0
MIN_LINE1_ZONE_REPAIR_GAIN = 4.0

PAIR_COUNTRY_MATCH_BONUS = 10.0
CANDIDATE_SUPPORT_BONUS_SCALE = 3.0

FAST_OCR_VARIANT_SOURCES = ("gray", "clahe")
FAST_OCR_VARIANT_SCALES = (2,)
FAST_OCR_VARIANT_THRESHOLDS = ("otsu",)
FAST_OCR_SPLIT_LABELS = ("projection", "half")
PADDLE_OCR_BACKEND_LABEL = "paddle"
_PADDLE_OCR_INSTANCE = None


# -------------------------------------------------------
# BASIC IO
# -------------------------------------------------------

def save(img, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[Save]  {path}")


# -------------------------------------------------------
# NORMALIZATION
# -------------------------------------------------------

def normalize_mrz(text: str) -> str:
    text = (text or "").upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9<]", "", text)
    text = text.replace("1ND", "IND")
    return text


def normalize_td3_line1(text: str) -> str:
    text = normalize_mrz(text)

    if not text.startswith("P<"):
        if text.startswith(("PO", "P0")):
            text = "P<" + text[2:]
        elif text.startswith("PK"):
            text = "P<" + text[2:]
        elif text.startswith("P"):
            text = "P<" + text[1:]

    text = text[:MRZ_LINE_LEN].ljust(MRZ_LINE_LEN, "<")
    return text


def normalize_td3_line2(text: str) -> str:
    text = normalize_mrz(text)
    text = text[:MRZ_LINE_LEN].ljust(MRZ_LINE_LEN, "<")
    return text


def _sanitize_alpha(text: str) -> str:
    return re.sub(r"[^A-Z]", "", normalize_mrz(text).replace("<", ""))


def _sanitize_name_token(token: str) -> str:
    return _sanitize_alpha(token)


def _sanitize_name_zone(value: str) -> str:
    return re.sub(r"[^A-Z<]", "", normalize_mrz(value))


def _split_tail_name_tokens(value: str) -> tuple[list[str], str]:
    zone = _sanitize_name_zone(value)
    trimmed = zone.strip("<")
    if not trimmed:
        return [], zone

    raw_parts = [part for part in trimmed.split("<") if part]
    tokens = [_sanitize_name_token(part) for part in raw_parts]
    tokens = [token for token in tokens if token]

    if not tokens:
        return [], zone

    rebuilt = "<" + "<".join(tokens)
    suffix_len = max(0, len(zone) - len(rebuilt))
    remainder = "<" * suffix_len
    return tokens, remainder


def _generate_country_code_variants(code: str) -> set[str]:
    code = normalize_mrz(code).replace("<", "")[:3]
    if len(code) != 3:
        return set()

    variants = {code}
    for i, ch in enumerate(code):
        for repl in COUNTRY_CODE_AMBIGUOUS_SUBS.get(ch, ()):
            candidate = code[:i] + repl + code[i + 1:]
            if re.fullmatch(r"[A-Z]{3}", candidate):
                variants.add(candidate)

    return variants


def _normalize_numeric_field(data: str) -> str:
    data = normalize_mrz(data)
    return "".join(NUMERIC_FIELD_SUBS.get(ch, ch) for ch in data)


def _ambiguous_doc_char_count(data: str) -> int:
    return sum(1 for ch in normalize_mrz(data) if ch in AMBIGUOUS_DOC_CHARS)


# -------------------------------------------------------
# CHECKSUM
# -------------------------------------------------------

def char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    return 0


def checksum(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(data):
        total += char_value(c) * weights[i % 3]
    return str(total % 10)


def _limited_field_variants(
    data: str,
    max_edits: int = 2,
    substitutions: dict[str, tuple[str, ...]] | None = None,
) -> set[str]:
    data = normalize_mrz(data)
    substitutions = substitutions or AMBIGUOUS_FIELD_SUBS
    variants = {data}

    pos = [i for i, ch in enumerate(data) if ch in substitutions]

    for i in pos:
        ch = data[i]
        for repl in substitutions[ch]:
            variants.add(data[:i] + repl + data[i + 1:])

    if max_edits >= 2 and len(pos) >= 2:
        for i, j in itertools.combinations(pos, 2):
            for repl_i in substitutions[data[i]]:
                for repl_j in substitutions[data[j]]:
                    chars = list(data)
                    chars[i] = repl_i
                    chars[j] = repl_j
                    variants.add("".join(chars))

    return variants


def _field_correction_key(candidate: str, original: str, field_kind: str) -> tuple:
    edits = sum(1 for src, dst in zip(original, candidate) if src != dst)

    if field_kind == "document_number":
        return (
            -_ambiguous_doc_char_count(candidate),
            int(bool(candidate) and candidate[0].isalpha()),
            -edits,
            candidate,
        )

    return (-edits, candidate)


def correct_field(
    data: str,
    check: str,
    *,
    substitutions: dict[str, tuple[str, ...]] | None = None,
    field_kind: str = "generic",
) -> str:
    data = normalize_mrz(data)
    check = normalize_mrz(check)[:1]

    if not data or not check or not check.isdigit():
        return data

    if checksum(data) == check and field_kind != "document_number":
        return data

    matches = [
        cand
        for cand in _limited_field_variants(data, max_edits=2, substitutions=substitutions)
        if checksum(cand) == check
    ]
    if not matches:
        return data

    return max(matches, key=lambda cand: _field_correction_key(cand, data, field_kind))


def validate_td3_checks(line2: str) -> dict:
    line2 = normalize_td3_line2(line2)
    if len(line2) < MRZ_LINE_LEN:
        return {
            "document": {"actual": "", "expected": "", "valid": False},
            "birth_date": {"actual": "", "expected": "", "valid": False},
            "expiry_date": {"actual": "", "expected": "", "valid": False},
            "personal_number": {"actual": "", "expected": "", "valid": False},
            "composite": {"actual": "", "expected": "", "valid": False},
            "passed_count": 0,
            "total_count": 5,
            "composite_valid": False,
        }

    doc_number = line2[0:9]
    doc_check = line2[9]
    dob = line2[13:19]
    dob_check = line2[19]
    expiry = line2[21:27]
    expiry_check = line2[27]
    personal = line2[28:42]
    personal_check = line2[42]
    final_check = line2[43]

    personal_expected = checksum(personal)
    personal_valid = personal_check == "<" and personal == "<" * 14
    if not personal_valid:
        personal_valid = personal_expected == personal_check

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]

    result = {
        "document": {
            "actual": doc_check,
            "expected": checksum(doc_number),
            "valid": checksum(doc_number) == doc_check,
        },
        "birth_date": {
            "actual": dob_check,
            "expected": checksum(dob),
            "valid": checksum(dob) == dob_check,
        },
        "expiry_date": {
            "actual": expiry_check,
            "expected": checksum(expiry),
            "valid": checksum(expiry) == expiry_check,
        },
        "personal_number": {
            "actual": personal_check,
            "expected": personal_expected,
            "valid": personal_valid,
        },
        "composite": {
            "actual": final_check,
            "expected": checksum(composite_data),
            "valid": checksum(composite_data) == final_check,
        },
    }

    passed = sum(1 for k in result.values() if k["valid"])
    result["passed_count"] = passed
    result["total_count"] = 5
    result["composite_valid"] = result["composite"]["valid"]
    return result


def validate_and_correct_mrz(line1: str, line2: str) -> tuple[str, str, dict]:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    if len(line2) < MRZ_LINE_LEN:
        return line1, line2, validate_td3_checks(line2)

    doc_number = correct_field(
        line2[0:9],
        line2[9],
        substitutions=DOC_NUMBER_AMBIGUOUS_SUBS,
        field_kind="document_number",
    )
    nationality = line2[10:13]
    valid_nationality_variants = [
        candidate
        for candidate in _generate_country_code_variants(nationality)
        if is_valid_mrz_country_code(candidate)
    ]
    if valid_nationality_variants:
        nationality = max(
            valid_nationality_variants,
            key=lambda candidate: (
                int(candidate == nationality),
                -sum(1 for src, dst in zip(nationality, candidate) if src != dst),
                candidate,
            ),
        )
    dob = correct_field(_normalize_numeric_field(line2[13:19]), line2[19])
    expiry = correct_field(_normalize_numeric_field(line2[21:27]), line2[27])
    personal = correct_field(line2[28:42], line2[42])

    repaired_line2 = (
        doc_number
        + line2[9]
        + nationality
        + dob
        + line2[19]
        + line2[20]
        + expiry
        + line2[27]
        + personal
        + line2[42]
        + line2[43]
    )

    checks = validate_td3_checks(repaired_line2)
    return line1, repaired_line2, checks


# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------

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


def _ocr_search_profile() -> dict:
    if FAST_OCR:
        return {
            "variant_sources": FAST_OCR_VARIANT_SOURCES,
            "variant_scales": FAST_OCR_VARIANT_SCALES,
            "variant_thresholds": FAST_OCR_VARIANT_THRESHOLDS,
            "split_labels": FAST_OCR_SPLIT_LABELS,
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
    }


def _prepare_variants(gray, prefix: str, profile: dict | None = None):
    profile = profile or _ocr_search_profile()
    variants = []

    variant_sources = profile["variant_sources"]
    base_images = {"gray": gray}
    if "clahe" in variant_sources:
        base_images["clahe"] = _apply_clahe(gray)

    for source_name in variant_sources:
        base = base_images[source_name]
        for scale in profile["variant_scales"]:
            scaled = _resize(base, scale)

            otsu = None
            for threshold_name in profile["variant_thresholds"]:
                if threshold_name == "otsu":
                    if otsu is None:
                        otsu = _otsu_thresh(scaled)
                    image = otsu
                    morph = "none"
                    threshold = "otsu"
                elif threshold_name == "adaptive":
                    image = _adaptive_thresh(scaled)
                    morph = "none"
                    threshold = "adaptive"
                elif threshold_name == "otsu_thick":
                    if otsu is None:
                        otsu = _otsu_thresh(scaled)
                    image = _thicken(otsu)
                    morph = "dilate"
                    threshold = "otsu"
                else:
                    continue

                variants.append({
                    "variant_id": f"{prefix}_{source_name}_s{scale}_{threshold_name}",
                    "image": image,
                    "meta": {
                        "scale": scale,
                        "source": source_name,
                        "threshold": threshold,
                        "morph": morph,
                    },
                })

    return variants


# -------------------------------------------------------
# LINE SPLIT
# -------------------------------------------------------

def split_mrz_lines(img):
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
    save(vis, "mrz_line_split.png")

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
    save(proj_vis, "mrz_line_projection.png")

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


def build_split_candidates(gray, base_meta: dict, profile: dict | None = None):
    profile = profile or _ocr_search_profile()
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


def estimate_ocr_search_space() -> dict:
    profile = _ocr_search_profile()
    per_line_variants = (
        len(profile["variant_sources"])
        * len(profile["variant_scales"])
        * len(profile["variant_thresholds"])
    )
    per_line_attempts = per_line_variants * len(OCR_CONFIGS)
    split_count = len(profile["split_labels"])

    return {
        "fast_ocr": FAST_OCR,
        "psms": list(TESSERACT_PSMS),
        "oems": list(TESSERACT_OEMS),
        "split_count": split_count,
        "per_line_variants": per_line_variants,
        "per_line_attempts": per_line_attempts,
        "total_tesseract_calls": split_count * 2 * per_line_attempts,
    }


# -------------------------------------------------------
# OCR
# -------------------------------------------------------

def _ocr_image(img, cfg: str) -> str:
    text = pytesseract.image_to_string(
        img,
        lang=TESSERACT_LANG,
        config=(
            f"{cfg} "
            f"-c tessedit_char_whitelist={MRZ_WHITELIST} "
            f"-c load_system_dawg=0 "
            f"-c load_freq_dawg=0"
        ),
    )
    return normalize_mrz(text)


def _resolve_ocr_backends() -> list[str]:
    backend = OCR_BACKEND
    if backend in {"tesseract", "paddle"}:
        return [backend]
    if backend in {"auto", "hybrid", "both"}:
        return ["tesseract", "paddle"]
    print(f"[WARN] Unknown OCR_BACKEND='{backend}', falling back to tesseract")
    return ["tesseract"]


def _resolve_paddle_use_gpu() -> bool:
    if not PADDLEOCR_USE_GPU:
        return False

    _ensure_stub_ccache_on_path()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            import paddle
        except ImportError:
            print("[WARN] Paddle GPU requested but 'paddle' is not installed")
            return False

    if not paddle.device.is_compiled_with_cuda():
        print("[WARN] Paddle GPU requested but the installed Paddle build has no CUDA support")
        return False

    try:
        device_count = paddle.device.cuda.device_count()
    except Exception as exc:
        print(f"[WARN] Paddle GPU requested but CUDA device detection failed: {exc}")
        return False

    if device_count < 1:
        print("[WARN] Paddle GPU requested but no CUDA devices are visible; falling back to CPU")
        return False

    return True


def _get_paddle_ocr():
    global _PADDLE_OCR_INSTANCE

    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE

    paddle_cache_home = _resolve_paddle_cache_home()
    os.environ["PADDLE_PDX_CACHE_HOME"] = paddle_cache_home
    os.makedirs(paddle_cache_home, exist_ok=True)
    _ensure_stub_ccache_on_path()
    if _should_disable_paddle_model_source_check(paddle_cache_home):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR backend requested but 'paddleocr' is not installed in the active environment"
            ) from exc

    use_gpu = _resolve_paddle_use_gpu()
    kwargs = {"lang": PADDLEOCR_LANG}
    signature = inspect.signature(PaddleOCR.__init__)

    if "use_gpu" in signature.parameters:
        kwargs["use_gpu"] = use_gpu
    if "show_log" in signature.parameters:
        kwargs["show_log"] = False
    if "use_doc_orientation_classify" in signature.parameters:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in signature.parameters:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in signature.parameters:
        kwargs["use_textline_orientation"] = False
    if "ocr_version" in signature.parameters:
        kwargs["ocr_version"] = "PP-OCRv5"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _PADDLE_OCR_INSTANCE = PaddleOCR(**kwargs)
    return _PADDLE_OCR_INSTANCE


def _extract_paddle_lines(result) -> list[str]:
    texts = []

    def walk(node):
        if node is None:
            return

        if isinstance(node, str):
            text = normalize_mrz(node)
            if text:
                texts.append(text)
            return

        if isinstance(node, dict):
            for key in ("rec_texts", "texts"):
                values = node.get(key)
                if isinstance(values, list):
                    for value in values:
                        walk(value)
                    return

            text = node.get("rec_text") or node.get("text")
            if isinstance(text, str):
                walk(text)
                return

            for value in node.values():
                walk(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)):
                if node[1] and isinstance(node[1][0], str):
                    walk(node[1][0])
                    return
            for value in node:
                walk(value)

    walk(result)
    return texts


def _trim_line1_spill(text: str) -> str:
    text = normalize_td3_line1(text)

    # Clean line-2 spill such as "...<<<<<<<<EK" that can appear when
    # PaddleOCR leaks the document-number prefix into line 1.
    text = re.sub(r"<{4,}[A-Z0-9]{1,6}$", lambda m: "<" * len(m.group(0)), text)

    if "<<" not in text:
        return text

    name_zone = text[5:]
    if any(ch.isdigit() for ch in name_zone):
        digits = sum(ch.isdigit() for ch in name_zone)
        text = normalize_td3_line1(re.sub(r"[0-9]", "<", text))
        if digits >= 2:
            return text

    return text


def _best_paddle_text_candidate(texts: list[str], line_kind: str) -> str:
    cleaned = [normalize_mrz(text) for text in texts if normalize_mrz(text)]
    if not cleaned:
        return ""

    candidates = []
    seen = set()

    def add_candidate(text: str) -> None:
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    for text in cleaned:
        add_candidate(text)
        for fragment in re.findall(r"[A-Z0-9<]{6,}", text):
            add_candidate(fragment)

    joined = normalize_mrz("".join(cleaned))
    add_candidate(joined)

    if line_kind == "line1":
        scored = []
        for cand in candidates:
            trimmed = _trim_line1_spill(cand)
            score = score_td3_line1(trimmed)
            if any(ch.isdigit() for ch in trimmed[5:]):
                score -= 80.0
            if re.search(r"<{4,}[A-Z0-9]{1,6}$", cand):
                score -= 40.0
            scored.append((score, trimmed))
        return max(scored, key=lambda item: item[0])[1]

    scored = []
    for cand in candidates:
        normalized = normalize_td3_line2(cand)
        score, checks = score_td3_line2(normalized)
        if not re.match(r"^[A-Z0-9<]{1,2}[A-Z0-9<]{7,}", normalized):
            score -= 20.0
        score += checks.get("passed_count", 0) * 5.0
        scored.append((score, normalized))

    return max(scored, key=lambda item: item[0])[1]


def _paddle_ocr_image(img, line_kind: str | None = None) -> str:
    ocr = _get_paddle_ocr()
    paddle_img = img

    # PaddleOCR expects an HWC image for array input; Stage 3 variants are often grayscale.
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

    texts = _extract_paddle_lines(result)
    if not texts:
        return ""

    if line_kind in {"line1", "line2"}:
        return _best_paddle_text_candidate(texts, line_kind)

    return normalize_mrz("".join(texts))


def generate_ocr_candidates(line_img, prefix: str):
    gray = _to_gray(line_img)
    variants = _prepare_variants(gray, prefix)
    backends = _resolve_ocr_backends()
    line_kind = "line1" if prefix.startswith("line1_") else "line2" if prefix.startswith("line2_") else None

    candidates = []
    seen = set()

    for v in variants:
        if "tesseract" in backends:
            for c in OCR_CONFIGS:
                text = _ocr_image(v["image"], c["cfg"])
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
            text = _paddle_ocr_image(v["image"], line_kind=line_kind)
            if text:
                norm = normalize_mrz(text)
                key = ("paddle", norm, v["variant_id"], PADDLE_OCR_BACKEND_LABEL)
                if key not in seen:
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


def _candidate_support_bonus(support_count: int) -> float:
    if support_count <= 1:
        return 0.0

    # Strong consensus across OCR variants is a high-signal indicator that the
    # text is real, especially for line 1 where noisy single-variant winners
    # can otherwise outrank the correct MRZ string.
    return (math.log2(support_count) ** 2) * CANDIDATE_SUPPORT_BONUS_SCALE


def _apply_candidate_support_bonus(candidates) -> None:
    support_counts = {}
    for cand in candidates:
        text = cand["text"]
        support_counts[text] = support_counts.get(text, 0) + 1

    for cand in candidates:
        support_count = support_counts[cand["text"]]
        support_bonus = _candidate_support_bonus(support_count)
        base_score = cand["score"]

        cand["base_score"] = base_score
        cand["support_count"] = support_count
        cand["support_bonus"] = support_bonus
        cand["score"] = base_score + support_bonus


# -------------------------------------------------------
# SCORING
# -------------------------------------------------------

def _max_consonant_run(token: str) -> int:
    vowels = set("AEIOUY")
    best = 0
    cur = 0
    for ch in token:
        if ch in vowels:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def _max_vowel_run(token: str) -> int:
    vowels = set("AEIOUY")
    best = 0
    cur = 0
    for ch in token:
        if ch in vowels:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _name_token_score(token: str) -> float:
    token = _sanitize_name_token(token)
    if not token:
        return -999.0

    score = 0.0
    n = len(token)
    vowels = sum(1 for c in token if c in "AEIOUY")

    if 3 <= n <= 8:
        score += 12
    elif 2 <= n <= 10:
        score += 6
    else:
        score -= abs(n - 5) * 2

    if vowels >= 2:
        score += 10
    elif vowels == 1:
        score += 2
    else:
        score -= 12

    if token[-1] in "AEIOUY":
        score += 5

    if len(token) >= 5 and token.endswith("K"):
        score -= 8

    if token.endswith("IA"):
        score += 6

    ugly_bigrams = {"DT", "TK", "KK", "KR", "RK", "RR", "RT", "TR", "XD", "XX", "QK", "KQ"}
    for i in range(len(token) - 1):
        bg = token[i:i + 2]
        if bg in ugly_bigrams:
            score -= 6

    score -= max(0, _max_consonant_run(token) - 2) * 6
    score -= max(0, _max_vowel_run(token) - 2) * 8

    for ch in set(token):
        count = token.count(ch)
        if count >= 3:
            score -= (count - 2) * 3

    return score


def score_td3_line1(text: str) -> float:
    text = normalize_td3_line1(text)
    score = 0.0

    if len(text) == MRZ_LINE_LEN:
        score += 20
    else:
        score -= abs(len(text) - MRZ_LINE_LEN) * 2

    if text.startswith("P<"):
        score += 20
    else:
        score -= 20

    issuing = text[2:5]
    if re.fullmatch(r"[A-Z<]{3}", issuing):
        score += 12
        if is_valid_mrz_country_code(issuing):
            score += 10
        else:
            score -= 8
    else:
        score -= 12

    name_zone = text[5:]
    if "<<" in name_zone:
        score += 10
        parts = name_zone.split("<<", 1)
        surname = _sanitize_name_token(parts[0])
        given_zone = parts[1]
        first_token_raw = given_zone.split("<", 1)[0]
        first_token = _sanitize_name_token(first_token_raw)
        filler_tail = given_zone[len(first_token_raw):]

        if surname:
            score += 8
        else:
            score -= 10

        if first_token:
            score += 8
            token_score = _name_token_score(first_token)
            score += max(-8, min(18, token_score))
            score += max(-4, min(6, token_score / 4.0))
        else:
            score -= 10

        tail_tokens, tail_remainder = _split_tail_name_tokens(filler_tail)
        for tail_token in tail_tokens:
            token_score = _name_token_score(tail_token)
            score += 6
            score += max(-6, min(14, token_score))
            if 2 <= len(tail_token) <= 12:
                score += 4

        tail_letters = sum(1 for c in tail_remainder if "A" <= c <= "Z")
        tail_fillers = filler_tail.count("<")
        score -= tail_letters * 2.5
        if tail_fillers >= max(6, len(filler_tail) // 2):
            score += 8
    else:
        score -= 20

    invalid = len(re.findall(r"[^A-Z<]", text))
    score -= invalid * 5

    return score


def score_td3_line2(text: str) -> tuple[float, dict]:
    text = normalize_td3_line2(text)
    score = 0.0
    doc_number = text[0:9]
    dob = text[13:19]
    expiry = text[21:27]

    if len(text) == MRZ_LINE_LEN:
        score += 20
    else:
        score -= abs(len(text) - MRZ_LINE_LEN) * 2

    if re.fullmatch(r"[A-Z0-9<]{44}", text):
        score += 10
    else:
        score -= 15

    score -= _ambiguous_doc_char_count(doc_number) * 3.0

    dob_non_digits = sum(1 for ch in dob if not ch.isdigit())
    expiry_non_digits = sum(1 for ch in expiry if not ch.isdigit())
    score -= (dob_non_digits + expiry_non_digits) * 10.0

    nationality = text[10:13]
    if re.fullmatch(r"[A-Z<]{3}", nationality):
        score += 10
        if is_valid_mrz_country_code(nationality):
            score += 10
        else:
            score -= 8
    else:
        score -= 10

    sex = text[20]
    if sex in {"M", "F", "<"}:
        score += 10
    else:
        score -= 12

    checks = validate_td3_checks(text)
    passed = checks["passed_count"]
    score += passed * 18

    return score, checks


def score_split_quality(split_info, best_line2_checks, best_line2_score):
    score = 0.0

    passed = best_line2_checks.get("passed_count", 0)
    score += passed * 40.0

    if best_line2_checks.get("composite_valid"):
        score += 25.0

    bottom_ratio = split_info["bottom_ratio"]

    if bottom_ratio < 0.28:
        score -= 25
    elif bottom_ratio < 0.33:
        score -= 10
    elif 0.35 <= bottom_ratio <= 0.55:
        score += 10

    score += best_line2_score
    return score


def pair_consistency_bonus(line1: str, line2: str) -> float:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    issuing_country = line1[2:5]
    nationality = line2[10:13]

    if (
        is_valid_mrz_country_code(issuing_country)
        and is_valid_mrz_country_code(nationality)
        and issuing_country == nationality
    ):
        return PAIR_COUNTRY_MATCH_BONUS

    return 0.0


# -------------------------------------------------------
# LINE-1 REPAIR
# -------------------------------------------------------

def _generate_token_variants(token: str, max_edits: int = 2) -> set[str]:
    token = _sanitize_name_token(token)
    if not token:
        return {token}

    variants = {token}
    positions = [i for i, ch in enumerate(token) if ch in TOKEN_AMBIGUOUS_SUBS]

    for i in positions:
        ch = token[i]
        for repl in TOKEN_AMBIGUOUS_SUBS[ch]:
            variants.add(token[:i] + repl + token[i + 1:])

    if max_edits >= 2 and len(positions) >= 2:
        for i, j in itertools.combinations(positions, 2):
            for repl_i in TOKEN_AMBIGUOUS_SUBS[token[i]]:
                for repl_j in TOKEN_AMBIGUOUS_SUBS[token[j]]:
                    chars = list(token)
                    chars[i] = repl_i
                    chars[j] = repl_j
                    variants.add("".join(chars))

    return {v for v in variants if v}


def _split_given_name_zone(given_raw: str) -> tuple[str, str]:
    raw_first, sep, raw_tail = given_raw.partition("<")
    given_token = _sanitize_name_token(raw_first)
    given_tail = _sanitize_name_zone(sep + raw_tail) if sep else ""
    return given_token, given_tail


def _build_td3_line1(
    issuing_country: str,
    surname: str,
    given_token: str,
    given_tail: str = "",
) -> str:
    issuing_country = re.sub(r"[^A-Z<]", "", (issuing_country or ""))
    issuing_country = issuing_country[:3].ljust(3, "<")
    surname = _sanitize_name_token(surname)
    given_token = _sanitize_name_token(given_token)
    given_tail = _sanitize_name_zone(given_tail)

    if given_tail and not given_tail.startswith("<"):
        given_tail = "<" + given_tail

    rebuilt = f"P<{issuing_country}{surname}<<{given_token}{given_tail}"
    return rebuilt[:MRZ_LINE_LEN].ljust(MRZ_LINE_LEN, "<")


def repair_given_name_token(token: str) -> tuple[str, dict]:
    raw = _sanitize_name_token(token)
    if not raw:
        return raw, {"changed": False, "from": token, "to": raw, "reason": "empty"}

    raw_score = _name_token_score(raw)

    candidates = {
        c for c in _generate_token_variants(raw, max_edits=2)
        if re.fullmatch(r"[A-Z]+", c)
    }
    if not candidates:
        candidates = {raw}
    ranked = sorted(
        ((cand, _name_token_score(cand)) for cand in candidates),
        key=lambda x: (x[1], -abs(len(x[0]) - 5), x[0][-1] in "AEIOUY"),
        reverse=True,
    )

    best, best_score = ranked[0]
    if best != raw and best_score < raw_score + MIN_TOKEN_REPAIR_SCORE_GAIN:
        best = raw
        best_score = raw_score

    changed = best != raw

    return best, {
        "changed": changed,
        "from": raw,
        "to": best,
        "reason": "token_ambiguity_repair",
        "candidates_considered": len(candidates),
        "best_score": best_score,
    }


def _build_filler_tail(length: int) -> str:
    if length <= 0:
        return ""
    return "<" * length


def _candidate_given_zone_repairs(given_raw: str) -> list[dict]:
    raw_first, sep, raw_tail = given_raw.partition("<")
    raw_token = _sanitize_name_token(raw_first)
    raw_tail_zone = _sanitize_name_zone(sep + raw_tail) if sep else ""

    candidates: list[dict] = [{
        "given_token": raw_token,
        "given_tail": raw_tail_zone,
        "reason": "baseline",
    }]

    if 3 <= len(raw_token) <= 12:
        repaired_token, meta = repair_given_name_token(raw_token)
        if meta["changed"]:
            candidates.append({
                "given_token": repaired_token,
                "given_tail": raw_tail_zone,
                "reason": "token_ambiguity_repair",
            })

    if len(raw_token) > 8:
        max_prefix_len = min(12, len(raw_token))
        for prefix_len in range(3, max_prefix_len + 1):
            prefix = raw_token[:prefix_len]
            suffix_len = len(raw_token) - prefix_len
            filler_tail = _build_filler_tail(suffix_len + len(raw_tail_zone))
            candidates.append({
                "given_token": prefix,
                "given_tail": filler_tail,
                "reason": "trim_long_given_token_noise",
            })

            repaired_prefix, meta = repair_given_name_token(prefix)
            if meta["changed"]:
                candidates.append({
                    "given_token": repaired_prefix,
                    "given_tail": filler_tail,
                    "reason": "trim_long_given_token_noise_with_token_repair",
                })

    deduped = {}
    for candidate in candidates:
        key = (candidate["given_token"], candidate["given_tail"])
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def repair_given_name_zone(
    issuing_country: str,
    surname: str,
    given_raw: str,
) -> tuple[str, str, dict | None]:
    baseline_token, baseline_tail = _split_given_name_zone(given_raw)
    baseline_line = _build_td3_line1(
        issuing_country,
        surname,
        baseline_token,
        baseline_tail,
    )
    baseline_score = score_td3_line1(baseline_line)

    ranked = []
    for candidate in _candidate_given_zone_repairs(given_raw):
        given_token = candidate["given_token"]
        given_tail = candidate["given_tail"]
        if not given_token:
            continue

        line = _build_td3_line1(issuing_country, surname, given_token, given_tail)
        score = score_td3_line1(line)
        ranked.append((score, given_token, given_tail, candidate["reason"]))

    if not ranked:
        return baseline_token, baseline_tail, None

    best_score, best_token, best_tail, best_reason = max(
        ranked,
        key=lambda item: (
            item[0],
            _name_token_score(item[1]),
            -len(item[2]),
            item[1].endswith("A"),
            item[1],
        ),
    )

    if (
        (best_token, best_tail) == (baseline_token, baseline_tail)
        or best_score < baseline_score + MIN_LINE1_ZONE_REPAIR_GAIN
    ):
        return baseline_token, baseline_tail, None

    return best_token, best_tail, {
        "field": "line1",
        "position": "given_name_zone",
        "from": f"{baseline_token}{baseline_tail}",
        "to": f"{best_token}{best_tail}",
        "reason": best_reason,
        "score_gain": round(best_score - baseline_score, 2),
    }


def repair_issuing_country_code(line1: str) -> tuple[str, dict | None]:
    line = normalize_td3_line1(line1)

    if not line.startswith("P<"):
        return line, None

    issuing_country = line[2:5]
    if is_valid_mrz_country_code(issuing_country):
        return line, None

    shifted_country = line[3:6]
    if is_valid_mrz_country_code(shifted_country):
        repaired = normalize_td3_line1(line[:2] + line[3:])
        return repaired, {
            "field": "line1",
            "position": "issuing_country",
            "from": issuing_country,
            "to": shifted_country,
            "reason": "drop_noise_before_country_code",
        }

    valid_variants = [
        candidate
        for candidate in _generate_country_code_variants(issuing_country)
        if is_valid_mrz_country_code(candidate)
    ]
    if not valid_variants:
        return line, None

    best_country = max(
        valid_variants,
        key=lambda candidate: (
            score_td3_line1(line[:2] + candidate + line[5:]),
            candidate,
        ),
    )
    repaired = normalize_td3_line1(line[:2] + best_country + line[5:])
    return repaired, {
        "field": "line1",
        "position": "issuing_country",
        "from": issuing_country,
        "to": best_country,
        "reason": "country_code_ambiguity_repair",
    }


def repair_td3_line1(line1: str) -> tuple[str, list[dict]]:
    repairs = []
    line = _trim_line1_spill(line1)

    if not line.startswith("P<"):
        return line, repairs

    line, country_repair = repair_issuing_country_code(line)
    if country_repair is not None:
        repairs.append(country_repair)

    issuing_country = line[2:5]
    name_zone = line[5:]

    if "<<" not in name_zone:
        return line, repairs

    surname_raw, given_raw = name_zone.split("<<", 1)
    surname = _sanitize_name_token(surname_raw)
    given_token, given_tail = _split_given_name_zone(given_raw)

    if not surname:
        return line, repairs

    if 3 <= len(surname) <= 10:
        repaired_surname, meta = repair_given_name_token(surname)
        if meta["changed"]:
            repairs.append({
                "field": "line1",
                "position": "surname",
                "from": surname,
                "to": repaired_surname,
                "reason": meta["reason"],
                "candidates_considered": meta["candidates_considered"],
                "best_score": meta["best_score"],
            })
            surname = repaired_surname

    repaired_given_token, repaired_given_tail, given_zone_repair = repair_given_name_zone(
        issuing_country,
        surname,
        given_raw,
    )
    if given_zone_repair is not None:
        given_token = repaired_given_token
        given_tail = repaired_given_tail
        repairs.append(given_zone_repair)

    rebuilt = _build_td3_line1(issuing_country, surname, given_token, given_tail)

    if rebuilt != line:
        repairs.append({
            "field": "line1",
            "position": "name",
            "from": name_zone,
            "to": f"{surname}<<{given_token}{given_tail}",
            "reason": "name_noise_collapse",
        })

    line = rebuilt

    if 3 <= len(given_token) <= 8:
        repaired_token, meta = repair_given_name_token(given_token)
        if meta["changed"]:
            rebuilt2 = _build_td3_line1(issuing_country, surname, repaired_token, given_tail)
            repairs.append({
                "field": "line1",
                "position": "given_name_token",
                "from": given_token,
                "to": repaired_token,
                "reason": meta["reason"],
                "candidates_considered": meta["candidates_considered"],
                "best_score": meta["best_score"],
            })
            line = rebuilt2

    return line, repairs


def _repair_paddle_line1_candidate(line1: str) -> tuple[str, dict | None]:
    line = _trim_line1_spill(line1)

    if not line.startswith("P<"):
        return line, None

    name_zone = line[5:]
    if "<<" not in name_zone:
        return line, None

    issuing_country = line[2:5]
    surname_raw, given_raw = name_zone.split("<<", 1)
    surname = _sanitize_name_token(surname_raw)

    if not surname:
        return line, None

    baseline_score = score_td3_line1(line)
    given_token, given_tail = _split_given_name_zone(given_raw)
    ranked = []
    surname_variants = {surname}
    for i, ch in enumerate(surname):
        if ch == "N":
            surname_variants.add(surname[:i] + "M" + surname[i + 1:])
        elif ch == "M":
            surname_variants.add(surname[:i] + "N" + surname[i + 1:])

    for variant in surname_variants:
        rebuilt = _build_td3_line1(issuing_country, variant, given_token, given_tail)
        ranked.append((
            score_td3_line1(rebuilt),
            variant.count("M") - surname.count("M"),
            surname.count("N") - variant.count("N"),
            variant,
            rebuilt,
        ))

    best_score, _, _, best_surname, best_line = max(ranked)
    if best_surname == surname or best_score < baseline_score:
        return line, None

    return best_line, {
        "field": "line1",
        "position": "surname",
        "from": surname,
        "to": best_surname,
        "reason": "paddle_surname_ambiguity_repair",
        "score_gain": round(best_score - baseline_score, 2),
    }


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


def run_ocr(mrz_img):
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
        f"psms={search_space['psms']} "
        f"splits={search_space['split_count']} "
        f"per_line_variants={search_space['per_line_variants']} "
        f"estimated_tesseract_calls={search_space['total_tesseract_calls']}"
    )

    split_candidates = build_split_candidates(gray, split_meta)
    all_line1_candidates = []
    all_line2_candidates = []
    split_summaries = []

    for split_info in split_candidates:
        split_y = split_info["split_y"]
        line1_img, line2_img = split_mrz_lines_at(gray, split_y)

        line1_candidates = generate_ocr_candidates(line1_img, f"line1_{split_info['label']}")
        line2_candidates = generate_ocr_candidates(line2_img, f"line2_{split_info['label']}")

        for cand in line1_candidates:
            text = _trim_line1_spill(cand["text"])
            repaired, repairs = repair_td3_line1(text)
            if cand.get("backend") == "paddle":
                paddle_repaired, paddle_meta = _repair_paddle_line1_candidate(repaired)
                if paddle_meta is not None:
                    repaired = paddle_repaired
                    repairs = repairs + [paddle_meta]
            cand["text"] = _trim_line1_spill(repaired)
            cand["repairs"] = repairs
            cand["score"] = score_td3_line1(cand["text"])
            cand["checksum_pass_count"] = None
            cand["split_label"] = split_info["label"]
            cand["split_y"] = split_info["split_y"]
            all_line1_candidates.append(cand)

        for cand in line2_candidates:
            text = normalize_td3_line2(cand["text"])
            _, repaired, checks = validate_and_correct_mrz("", text)
            cand["text"] = repaired
            cand["checks"] = checks
            cand["score"] = score_td3_line2(repaired)[0]
            cand["checksum_pass_count"] = checks["passed_count"]
            cand["split_label"] = split_info["label"]
            cand["split_y"] = split_info["split_y"]
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

    if not all_line1_candidates or not all_line2_candidates:
        raise RuntimeError("No valid split candidates produced OCR results")

    _apply_candidate_support_bonus(all_line1_candidates)

    line1_ranked = _rank_candidates(all_line1_candidates, "score")
    line2_ranked = _rank_candidates(all_line2_candidates, "score")
    line1_top = line1_ranked[:5]
    line2_top = line2_ranked[:5]

    best_pair = None
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

            candidate_key = _pair_selection_key(candidate_pair)
            best_key = None
            if best_pair is not None:
                best_key = _pair_selection_key(best_pair)

            if best_pair is None or candidate_key > best_key:
                best_pair = candidate_pair

    if best_pair is None:
        raise RuntimeError("No valid line pair selected from OCR candidates")

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
            "line1_top": [
                {
                    "text": c["text"],
                    "score": round(c["score"], 2),
                    "base_score": round(c.get("base_score", c["score"]), 2),
                    "variant_id": c["variant_id"],
                    "psm": c["psm"],
                    "checksum_pass_count": c["checksum_pass_count"],
                    "support_count": c.get("support_count", 1),
                    "support_bonus": round(c.get("support_bonus", 0.0), 2),
                    "split_label": c.get("split_label"),
                }
                for c in line1_top
            ],
            "line2_top": [
                {
                    "text": c["text"],
                    "score": round(c["score"], 2),
                    "variant_id": c["variant_id"],
                    "psm": c["psm"],
                    "checksum_pass_count": c["checksum_pass_count"],
                    "split_label": c.get("split_label"),
                }
                for c in line2_top
            ],
        },
        "selected": {
            "line1": {
                "score": round(best_pair["line1"]["score"], 2),
                "base_score": round(
                    best_pair["line1"].get("base_score", best_pair["line1"]["score"]),
                    2,
                ),
                "text": best_line1,
                "variant_id": best_pair["line1"]["variant_id"],
                "psm": best_pair["line1"]["psm"],
                "split_label": selected_line1_split,
                "variant_meta": best_pair["line1"]["variant_meta"],
                "support_count": best_pair["line1"].get("support_count", 1),
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
        "repairs_applied": best_pair["repairs_applied"],
        "parsed_fields": parsed,
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
