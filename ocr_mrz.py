import itertools
import os
import re
from datetime import datetime

import cv2
import numpy as np
import pytesseract

from env_utils import load_env_file


load_env_file()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# -------------------------------------------------------
# TESSERACT PATH (Windows)
# -------------------------------------------------------

paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

env_tesseract_cmd = os.getenv("TESSERACT_CMD")

if env_tesseract_cmd and os.path.isfile(env_tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = env_tesseract_cmd
elif env_tesseract_cmd:
    print(f"[WARN] TESSERACT_CMD is set but file not found: {env_tesseract_cmd}")

if (
    not getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    or not os.path.isfile(pytesseract.pytesseract.tesseract_cmd)
):
    for p in paths:
        if os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_LINE_LEN = 44
PSM_MODES = (7, 6, 13)

AMBIGUOUS_SUBS = {
    "O": "0", "0": "O",
    "I": "1", "1": "I",
    "Z": "2", "2": "Z",
    "S": "5", "5": "S",
    "B": "8", "8": "B",
    "G": "6", "6": "G",
    "Q": "0",
}
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}
# -------------------------------------------------------
# LINE-1 TOKEN MICRO-REPAIR
# -------------------------------------------------------

TOKEN_AMBIGUOUS_SUBS = {
    "T": ("I",),
    "I": ("T", "1", "L"),
    "L": ("I",),
    "1": ("I", "L"),
    "0": ("O",),
    "O": ("0",),
}

def save(img, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[Save]  {path}")


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise RuntimeError("Cannot load MRZ image")
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _apply_clahe(gray: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _ensure_binary_polarity(bin_img: np.ndarray) -> np.ndarray:
    # Tesseract is usually more stable when the page background is white.
    black_ratio = float(np.mean(bin_img < 128))
    if black_ratio > 0.65:
        return cv2.bitwise_not(bin_img)
    return bin_img


# -------------------------------------------------------
# NORMALIZE MRZ TEXT
# -------------------------------------------------------

def normalize_mrz(text):
    text = (text or "").upper()
    text = text.replace(" ", "").replace("\n", "").replace("\r", "")
    text = text.replace("«", "<")
    text = re.sub(r"[^A-Z0-9<]", "", text)
    text = text.replace("1ND", "IND")
    return text


def _enforce_td3_length(text: str) -> str:
    return text[:MRZ_LINE_LEN].ljust(MRZ_LINE_LEN, "<")


def _to_digit(c: str) -> str:
    if c.isdigit():
        return c
    return LETTER_TO_DIGIT.get(c, c)


def _to_letter(c: str) -> str:
    if "A" <= c <= "Z":
        return c
    return DIGIT_TO_LETTER.get(c, c)


def normalize_td3_line1(text: str) -> str:
    line = _enforce_td3_length(normalize_mrz(text))
    chars = list(line)

    # TD3 structure: position 0 document type, 1 filler, 2:5 issuing country.
    chars[0] = _to_letter(chars[0])
    chars[1] = "<" if chars[1] != "<" else chars[1]
    for i in range(2, 5):
        chars[i] = _to_letter(chars[i])

    for i in range(5, MRZ_LINE_LEN):
        chars[i] = _to_letter(chars[i]) if chars[i].isdigit() else chars[i]

    return "".join(chars)


def normalize_td3_line2(text: str) -> str:
    line = _enforce_td3_length(normalize_mrz(text))
    chars = list(line)

    for i in range(10, 13):
        chars[i] = _to_letter(chars[i])

    numeric_positions = set(range(13, 20)) | set(range(21, 28)) | {9, 42, 43}
    for idx in numeric_positions:
        if idx == 42 and chars[idx] == "<":
            continue
        chars[idx] = _to_digit(chars[idx])

    if chars[20] not in ("M", "F", "<"):
        chars[20] = "<"

    return "".join(chars)


# -------------------------------------------------------
# ICAO CHECKSUM
# -------------------------------------------------------

def char_value(c):
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    return 0


def checksum(data):
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(data):
        total += char_value(c) * weights[i % 3]
    return str(total % 10)


def _valid_yymmdd(value: str) -> bool:
    if not re.fullmatch(r"\d{6}", value):
        return False
    yy = int(value[0:2])
    mm = int(value[2:4])
    dd = int(value[4:6])
    if mm < 1 or mm > 12:
        return False
    if dd < 1:
        return False

    # Leap-year-safe day validation by anchoring in 2000+yy.
    year = 2000 + yy
    try:
        datetime(year, mm, dd)
        return True
    except ValueError:
        return False


def validate_td3_checks(line2: str) -> dict:
    line2 = normalize_td3_line2(line2)

    doc_number = line2[0:9]
    doc_check = line2[9]
    dob = line2[13:19]
    dob_check = line2[19]
    expiry = line2[21:27]
    expiry_check = line2[27]
    personal = line2[28:42]
    personal_check = line2[42]
    final_check = line2[43]

    doc_expected = checksum(doc_number)
    dob_expected = checksum(dob)
    expiry_expected = checksum(expiry)
    personal_expected = checksum(personal)

    if personal == "<" * 14 and personal_check == "<":
        personal_valid = True
    else:
        personal_valid = personal_check == personal_expected

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    composite_expected = checksum(composite_data)

    checks = {
        "document": {
            "actual": doc_check,
            "expected": doc_expected,
            "valid": doc_check == doc_expected,
        },
        "birth_date": {
            "actual": dob_check,
            "expected": dob_expected,
            "valid": dob_check == dob_expected,
        },
        "expiry_date": {
            "actual": expiry_check,
            "expected": expiry_expected,
            "valid": expiry_check == expiry_expected,
        },
        "personal_number": {
            "actual": personal_check,
            "expected": personal_expected,
            "valid": personal_valid,
        },
        "composite": {
            "actual": final_check,
            "expected": composite_expected,
            "valid": final_check == composite_expected,
        },
    }

    passed = sum(1 for item in checks.values() if item["valid"])
    checks["passed_count"] = passed
    checks["total_count"] = 5
    checks["composite_valid"] = checks["composite"]["valid"]
    return checks

# -------------------------------------------------------
# OCR PREPROCESSING
# -------------------------------------------------------

def clean_mrz_image(mrz_img):
    """
    Legacy-compatible "single best clean" image used by run_pipeline for
    preview/debug output. Multi-variant OCR preprocessing is done in run_ocr().
    """
    gray = _ensure_gray(mrz_img)
    clahe = _apply_clahe(gray)
    resized = cv2.resize(clahe, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=40, sigmaSpace=40)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = _ensure_binary_polarity(otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean


def _save_global_preprocess_debug(gray: np.ndarray) -> None:
    clahe = _apply_clahe(gray)
    scaled = cv2.resize(clahe, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(scaled, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        scaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )

    save(gray, "mrz_gray.png")
    save(clahe, "mrz_clahe.png")
    save(_ensure_binary_polarity(otsu), "mrz_thresh_otsu.png")
    save(_ensure_binary_polarity(adaptive), "mrz_thresh_adaptive.png")


def _projection_plot(profile: np.ndarray, split_y: int, reliable: bool) -> np.ndarray:
    h = 220
    w = max(len(profile), 220)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    max_val = float(np.max(profile)) if np.max(profile) > 0 else 1.0
    prev = None
    for x in range(len(profile)):
        v = profile[x] / max_val
        y = h - 10 - int(v * (h - 25))
        pt = (x, y)
        if prev is not None:
            cv2.line(canvas, prev, pt, (40, 40, 40), 1)
        prev = pt

    color = (0, 170, 0) if reliable else (0, 0, 255)
    split_x = int(np.clip(split_y, 0, w - 1))
    cv2.line(canvas, (split_x, 0), (split_x, h - 1), color, 2)
    cv2.putText(
        canvas,
        f"split={split_y} ({'profile' if reliable else 'fallback'})",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def split_mrz_lines(mrz_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    gray = _ensure_gray(mrz_img)
    clahe = _apply_clahe(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    binary_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    profile = np.sum(binary_inv > 0, axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()

    h = gray.shape[0]
    center = h // 2
    low = int(h * 0.25)
    high = int(h * 0.75)
    if high <= low:
        low, high = 0, h

    search = smooth[low:high]
    split_y = center
    reliable = False
    if search.size > 0:
        local_min_idx = int(np.argmin(search))
        split_y = low + local_min_idx

        top_peak = float(np.max(smooth[: max(split_y, 1)]))
        bot_peak = float(np.max(smooth[min(split_y + 1, h - 1):]))
        valley = float(smooth[split_y])

        min_peak = min(top_peak, bot_peak)
        ratio = valley / (min_peak + 1e-6)
        reliable = min_peak > 3 and ratio < 0.62 and int(h * 0.28) <= split_y <= int(h * 0.72)

    if not reliable:
        split_y = center

    split_y = int(np.clip(split_y, int(h * 0.2), int(h * 0.8)))
    line1 = gray[:split_y, :]
    line2 = gray[split_y:, :]

    if line1.size == 0 or line2.size == 0:
        split_y = center
        line1 = gray[:split_y, :]
        line2 = gray[split_y:, :]
        reliable = False

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(
        overlay,
        (0, split_y),
        (overlay.shape[1] - 1, split_y),
        (0, 255, 0) if reliable else (0, 0, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"split_y={split_y} ({'profile' if reliable else 'fallback-half'})",
        (6, max(20, split_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if reliable else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    save(overlay, "mrz_line_split.png")
    save(_projection_plot(smooth, split_y, reliable), "mrz_line_projection.png")
    save(line1, "mrz_line1.png")
    save(line2, "mrz_line2.png")

    return line1, line2, {
        "split_y": split_y,
        "method": "projection_valley" if reliable else "half_fallback",
        "profile_reliable": reliable,
        "image_height": h,
    }


def _line_variants(line_img: np.ndarray, line_name: str) -> list[dict]:
    gray = _ensure_gray(line_img)
    clahe = _apply_clahe(gray)
    variants = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for source_name, source_img in (("gray", gray), ("clahe", clahe)):
        for scale in (2, 3, 4):
            resized = cv2.resize(
                source_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            denoised = cv2.medianBlur(resized, 3)
            if scale >= 3:
                denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=35, sigmaSpace=35)

            blur = cv2.GaussianBlur(denoised, (3, 3), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                12,
            )

            otsu = _ensure_binary_polarity(otsu)
            adaptive = _ensure_binary_polarity(adaptive)
            thick = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
            thick = cv2.dilate(thick, kernel, iterations=1)

            variants.append(
                {
                    "id": f"{line_name}_{source_name}_s{scale}_otsu",
                    "image": otsu,
                    "scale": scale,
                    "source": source_name,
                    "threshold": "otsu",
                    "morph": "none",
                }
            )
            variants.append(
                {
                    "id": f"{line_name}_{source_name}_s{scale}_adaptive",
                    "image": adaptive,
                    "scale": scale,
                    "source": source_name,
                    "threshold": "adaptive",
                    "morph": "none",
                }
            )
            variants.append(
                {
                    "id": f"{line_name}_{source_name}_s{scale}_otsu_thick",
                    "image": thick,
                    "scale": scale,
                    "source": source_name,
                    "threshold": "otsu",
                    "morph": "close_dilate",
                }
            )

    return variants

# -------------------------------------------------------
# SCORING / REPAIR
# -------------------------------------------------------

def _longest_alpha_run(text: str) -> int:
    longest = 0
    for match in re.finditer(r"[A-Z]+", text):
        longest = max(longest, len(match.group(0)))
    return longest


def analyze_td3_line1_given_prefix(name_zone: str) -> dict:
    """
    Analyze the TD3 line1 name zone (positions 5:44) and estimate the most
    plausible given-name prefix after the first '<<' delimiter.
    """
    info = {
        "delimiter_found": False,
        "delimiter_index": -1,
        "prefix": "",
        "prefix_start": -1,
        "prefix_end": -1,
        "prefix_len": 0,
        "tail_starts_filler": False,
        "first_filler_index_in_tail": -1,
        "trailing_letters": 0,
        "trailing_noise_letters": 0,
        "trailing_noise_ratio": 0.0,
        "longest_trailing_alpha_run": 0,
        "clean_secondary_token": False,
        "hallucinated_tail": False,
    }

    delimiter_index = name_zone.find("<<")
    if delimiter_index < 0:
        return info

    info["delimiter_found"] = True
    info["delimiter_index"] = delimiter_index

    given_zone = name_zone[delimiter_index + 2:]
    leading_fillers = len(given_zone) - len(given_zone.lstrip("<"))
    core = given_zone[leading_fillers:]
    if not core:
        return info

    lead = re.match(r"^[A-Z]+", core)
    if not lead:
        return info

    lead_letters = lead.group(0)
    upper = min(len(lead_letters), 16)
    lower = 1 if upper == 1 else 2

    best = None
    for pref_len in range(lower, upper + 1):
        prefix = core[:pref_len]
        tail = core[pref_len:]

        first_filler = tail.find("<")
        trailing_letters = sum(1 for c in tail if "A" <= c <= "Z")
        trailing_noise = sum(1 for c in tail if c in "KRETX")
        noise_ratio = (trailing_noise / trailing_letters) if trailing_letters else 0.0
        longest_tail_run = _longest_alpha_run(tail)

        vowels = sum(1 for c in prefix if c in "AEIOUY")
        vowel_ratio = vowels / len(prefix) if prefix else 0.0

        token_score = 0.0
        if 2 <= pref_len <= 10:
            token_score += 5.0
        elif pref_len <= 14:
            token_score += 2.0
        else:
            token_score -= 1.0
        token_score -= 0.45 * max(0, pref_len - 8)

        if tail.startswith("<"):
            token_score += 7.0
        if first_filler >= 0:
            token_score += max(0.0, 3.5 - (0.8 * first_filler))
        else:
            token_score -= 6.0

        token_score -= 0.9 * trailing_letters
        token_score -= 5.5 * noise_ratio
        if longest_tail_run >= 6:
            token_score -= float(longest_tail_run - 5)
        if re.search(r"<{8,}", tail):
            token_score += 2.5

        if 0.25 <= vowel_ratio <= 0.8:
            token_score += 1.5
        elif pref_len >= 4:
            token_score -= 1.5

        if prefix[-1] in "AEIOUY":
            token_score += 0.8
        if prefix[-1] in "KRXQ":
            token_score -= 1.2

        pick = (token_score, -trailing_letters, -pref_len)
        if best is None or pick > best["pick"]:
            best = {
                "pick": pick,
                "prefix": prefix,
                "prefix_len": pref_len,
                "tail": tail,
                "first_filler": first_filler,
                "trailing_letters": trailing_letters,
                "trailing_noise": trailing_noise,
                "noise_ratio": noise_ratio,
                "longest_tail_run": longest_tail_run,
            }

    if best is None:
        return info

    prefix_start = delimiter_index + 2 + leading_fillers
    prefix_end = prefix_start + best["prefix_len"]
    tail = best["tail"]

    clean_secondary = bool(re.match(r"^<[A-Z]{2,}(?:<[A-Z]{2,})*<{4,}$", tail))
    hallucinated_tail = (
        best["trailing_letters"] >= 3
        and not clean_secondary
        and (
            best["noise_ratio"] >= 0.55
            or best["longest_tail_run"] >= 6
            or (best["first_filler"] < 0 and best["trailing_letters"] >= 5)
        )
    )

    info.update(
        {
            "prefix": best["prefix"],
            "prefix_start": prefix_start,
            "prefix_end": prefix_end,
            "prefix_len": best["prefix_len"],
            "tail_starts_filler": tail.startswith("<"),
            "first_filler_index_in_tail": best["first_filler"],
            "trailing_letters": best["trailing_letters"],
            "trailing_noise_letters": best["trailing_noise"],
            "trailing_noise_ratio": best["noise_ratio"],
            "longest_trailing_alpha_run": best["longest_tail_run"],
            "clean_secondary_token": clean_secondary,
            "hallucinated_tail": hallucinated_tail,
        }
    )
    return info


def collapse_td3_line1_given_noise(name_zone: str, analysis: dict) -> tuple[str, bool]:
    """
    Collapse random alphabetic garbage after the plausible given-name prefix
    into MRZ filler characters '<'.
    """
    if not analysis.get("delimiter_found"):
        return name_zone, False
    if not analysis.get("prefix"):
        return name_zone, False

    prefix_end = analysis.get("prefix_end", -1)
    if prefix_end <= 0 or prefix_end >= len(name_zone):
        return name_zone, False

    tail = name_zone[prefix_end:]
    trailing_letters = sum(1 for c in tail if "A" <= c <= "Z")
    trailing_noise = sum(1 for c in tail if c in "KRETX")
    noise_ratio = (trailing_noise / trailing_letters) if trailing_letters else 0.0

    should_collapse = False
    if analysis.get("hallucinated_tail"):
        should_collapse = True
    elif trailing_letters >= 8 and not analysis.get("clean_secondary_token"):
        should_collapse = True
    elif (
        trailing_letters >= 4
        and not analysis.get("tail_starts_filler")
        and noise_ratio >= 0.45
        and not analysis.get("clean_secondary_token")
    ):
        should_collapse = True

    if not should_collapse:
        return name_zone, False

    cleaned_tail = re.sub(r"[A-Z0-9]", "<", tail)
    if cleaned_tail == tail:
        return name_zone, False
    return name_zone[:prefix_end] + cleaned_tail, True


def score_td3_line1(line1: str) -> tuple[float, dict]:
    line = normalize_td3_line1(line1)
    score = 0.0
    details = {}

    if len(line) == MRZ_LINE_LEN:
        score += 24
    else:
        score -= abs(len(line) - MRZ_LINE_LEN) * 2
    details["length"] = len(line)

    invalid_count = sum(1 for c in line if c not in MRZ_WHITELIST)
    score += 16 if invalid_count == 0 else -8 * invalid_count
    details["invalid_chars"] = invalid_count

    if line.startswith("P<"):
        score += 20
    elif line[0] == "P":
        score += 10
    details["prefix"] = line[:2]

    country = line[2:5]
    if re.fullmatch(r"[A-Z]{3}", country):
        score += 10
    elif re.fullmatch(r"[A-Z<]{3}", country):
        score += 4
    details["country"] = country

    name_zone = line[5:]
    if "<<" in name_zone:
        score += 12
    else:
        score -= 8

    parts = name_zone.split("<<", 1)
    surname_ok = bool(parts and re.search(r"[A-Z]", parts[0]))
    given_ok = bool(len(parts) > 1 and re.search(r"[A-Z]", parts[1]))
    if surname_ok:
        score += 4
    if given_ok:
        score += 4

    given_analysis = analyze_td3_line1_given_prefix(name_zone)
    prefix = given_analysis["prefix"]
    prefix_len = given_analysis["prefix_len"]
    trailing_letters = given_analysis["trailing_letters"]
    trailing_noise = given_analysis["trailing_noise_letters"]

    if prefix_len >= 2:
        score += 6
    if 2 <= prefix_len <= 10:
        score += 7
    elif prefix_len > 12:
        score -= (prefix_len - 12) * 0.8

    if given_analysis["tail_starts_filler"]:
        score += 14
    elif given_analysis["delimiter_found"] and prefix_len > 0:
        score -= 4

    score -= 1.4 * trailing_letters
    if trailing_letters:
        score -= 5.0 * given_analysis["trailing_noise_ratio"]

    if given_analysis["longest_trailing_alpha_run"] >= 6:
        score -= (given_analysis["longest_trailing_alpha_run"] - 5) * 1.5

    if trailing_letters == 0 and prefix_len >= 2:
        score += 8
    if given_analysis["clean_secondary_token"]:
        score += 2.5
    if given_analysis["hallucinated_tail"]:
        score -= 8

    if prefix:
        vowels = sum(1 for c in prefix if c in "AEIOUY")
        vowel_ratio = vowels / len(prefix)
        if 0.25 <= vowel_ratio <= 0.8:
            score += 2
        else:
            score -= 1.5
        if prefix[-1] in "AEIOUY":
            score += 1.2
        if prefix[-1] in "KRXQ":
            score -= 1.8
        if re.search(r"[BCDFGHJKLMNPQRSTVWXZ]{4,}", prefix):
            score -= 2.5
        details["given_prefix_vowel_ratio"] = round(vowel_ratio, 3)

    details["has_delimiter"] = "<<" in name_zone
    details["surname_like"] = surname_ok
    details["given_like"] = given_ok
    details["given_prefix"] = prefix
    details["given_prefix_len"] = prefix_len
    details["given_tail_starts_filler"] = given_analysis["tail_starts_filler"]
    details["given_trailing_letters"] = trailing_letters
    details["given_trailing_noise_letters"] = trailing_noise
    details["given_trailing_noise_ratio"] = round(given_analysis["trailing_noise_ratio"], 3)
    details["given_hallucinated_tail"] = given_analysis["hallucinated_tail"]

    return score, details


def score_td3_line2(line2: str) -> tuple[float, dict]:
    line = normalize_td3_line2(line2)
    score = 0.0
    details = {}

    if len(line) == MRZ_LINE_LEN:
        score += 24
    else:
        score -= abs(len(line) - MRZ_LINE_LEN) * 2
    details["length"] = len(line)

    invalid_count = sum(1 for c in line if c not in MRZ_WHITELIST)
    score += 16 if invalid_count == 0 else -8 * invalid_count
    details["invalid_chars"] = invalid_count

    nationality = line[10:13]
    if re.fullmatch(r"[A-Z]{3}", nationality):
        score += 8
    elif re.fullmatch(r"[A-Z<]{3}", nationality):
        score += 3
    details["nationality"] = nationality

    sex = line[20]
    if sex in ("M", "F", "<"):
        score += 6
    details["sex"] = sex

    dob = line[13:19]
    exp = line[21:27]
    if _valid_yymmdd(dob):
        score += 8
    if _valid_yymmdd(exp):
        score += 8
    details["dob_valid"] = _valid_yymmdd(dob)
    details["expiry_valid"] = _valid_yymmdd(exp)

    checks = validate_td3_checks(line)
    score += checks["passed_count"] * 12
    if checks["composite_valid"]:
        score += 10
    details["checks_passed"] = checks["passed_count"]
    details["composite_valid"] = checks["composite_valid"]

    return score, details


def _field_search_to_match_check(
    data: str,
    target_check: str,
    allowed_chars: str,
    max_changes: int,
    field_name: str,
) -> list[dict]:
    if checksum(data) == target_check and all(c in allowed_chars for c in data):
        return [{"data": data, "edits": 0, "repairs": []}]

    mutable = []
    for i, c in enumerate(data):
        mapped = AMBIGUOUS_SUBS.get(c)
        if mapped and mapped in allowed_chars and mapped != c:
            mutable.append((i, mapped))

    out = []
    mutable_indices = [item[0] for item in mutable]
    lookup = {idx: mapped for idx, mapped in mutable}

    for edit_count in range(1, max_changes + 1):
        for combo in itertools.combinations(mutable_indices, edit_count):
            chars = list(data)
            repairs = []
            for idx in combo:
                original = chars[idx]
                chars[idx] = lookup[idx]
                repairs.append(
                    {
                        "field": field_name,
                        "position": idx,
                        "from": original,
                        "to": chars[idx],
                        "reason": "ambiguous_substitution",
                    }
                )
            candidate = "".join(chars)
            if not all(ch in allowed_chars for ch in candidate):
                continue
            if checksum(candidate) == target_check:
                out.append({"data": candidate, "edits": edit_count, "repairs": repairs})
        if out:
            # Stop at minimal edit distance.
            break
    return out


def _repair_checksum_field(
    field_name: str,
    data: str,
    check_char: str,
    allowed_chars: str,
    max_changes: int,
    allow_blank_check: bool = False,
) -> dict:
    candidates = []
    original_check = check_char
    check_options = set()

    if check_char.isdigit():
        check_options.add(check_char)
    if allow_blank_check and check_char == "<":
        check_options.add("<")
    mapped = LETTER_TO_DIGIT.get(check_char)
    if mapped:
        check_options.add(mapped)

    if allow_blank_check and data == "<" * len(data):
        expected_from_data = "<"
    else:
        expected_from_data = checksum(data)

    # Option A: keep check (or mapped check) and correct the data to match it.
    for check_opt in check_options:
        if check_opt == "<":
            continue
        matches = _field_search_to_match_check(
            data,
            check_opt,
            allowed_chars,
            max_changes=max_changes,
            field_name=field_name,
        )
        for m in matches:
            check_change = 0.0 if check_opt == original_check else 1.2
            repairs = list(m["repairs"])
            if check_opt != original_check:
                repairs.append(
                    {
                        "field": field_name,
                        "position": "check",
                        "from": original_check,
                        "to": check_opt,
                        "reason": "check_digit_normalized",
                    }
                )
            candidates.append(
                {
                    "data": m["data"],
                    "check": check_opt,
                    "cost": float(m["edits"]) + check_change,
                    "edits": m["edits"],
                    "repairs": repairs,
                }
            )

    # Option B: keep data and adjust check to computed checksum.
    check_repair_cost = 0.0 if expected_from_data == original_check else 1.2
    check_repairs = []
    if expected_from_data != original_check:
        check_repairs.append(
            {
                "field": field_name,
                "position": "check",
                "from": original_check,
                "to": expected_from_data,
                "reason": "check_digit_recomputed",
            }
        )
    candidates.append(
        {
            "data": data,
            "check": expected_from_data,
            "cost": check_repair_cost,
            "edits": 0,
            "repairs": check_repairs,
        }
    )

    # Option C: if data already matches original check, keep as-is.
    if original_check.isdigit() and checksum(data) == original_check:
        candidates.append(
            {"data": data, "check": original_check, "cost": 0.0, "edits": 0, "repairs": []}
        )
    if allow_blank_check and original_check == "<" and data == "<" * len(data):
        candidates.append(
            {"data": data, "check": "<", "cost": 0.0, "edits": 0, "repairs": []}
        )

    candidates.sort(
        key=lambda x: (
            x["cost"],
            len(x["repairs"]),
            x["edits"],
            0 if x["check"] == original_check else 1,
        )
    )
    return candidates[0]

def repair_td3_line1(line1: str, fallback_country: str | None = None) -> tuple[str, list]:
    line = normalize_td3_line1(line1)
    repairs = []
    chars = list(line)

    if chars[0] != "P":
        repairs.append(
            {"field": "line1", "position": 0, "from": chars[0], "to": "P", "reason": "force_document_type"}
        )
        chars[0] = "P"
    if chars[1] != "<":
        repairs.append(
            {"field": "line1", "position": 1, "from": chars[1], "to": "<", "reason": "td3_prefix_filler"}
        )
        chars[1] = "<"

    for i in range(2, 5):
        mapped = _to_letter(chars[i])
        if mapped != chars[i]:
            repairs.append(
                {"field": "line1", "position": i, "from": chars[i], "to": mapped, "reason": "country_letter_fix"}
            )
            chars[i] = mapped

    country = "".join(chars[2:5])
    if not re.fullmatch(r"[A-Z]{3}", country) and fallback_country and re.fullmatch(r"[A-Z]{3}", fallback_country):
        repairs.append(
            {"field": "line1", "position": "2:5", "from": country, "to": fallback_country, "reason": "country_fallback"}
        )
        chars[2:5] = list(fallback_country)

    name_zone = "".join(chars[5:])
    name_zone = "".join(_to_letter(c) if c.isdigit() else c for c in name_zone)

    if "<<" not in name_zone:
        multi = re.search(r"<{2,}", name_zone)
        if multi:
            s, e = multi.span()
            repaired = name_zone[:s] + "<<" + name_zone[e:]
            repairs.append(
                {"field": "line1", "position": "name", "from": name_zone, "to": repaired, "reason": "insert_name_delimiter"}
            )
            name_zone = repaired
        else:
            single = name_zone.find("<")
            if single >= 0:
                repaired = name_zone[:single] + "<<" + name_zone[single + 1:]
                repairs.append(
                    {"field": "line1", "position": "name", "from": name_zone, "to": repaired, "reason": "upgrade_single_filler_to_delimiter"}
                )
                name_zone = repaired

    first_delim = name_zone.find("<<")
    if first_delim >= 0:
        analysis = analyze_td3_line1_given_prefix(name_zone)
        repaired, collapsed = collapse_td3_line1_given_noise(name_zone, analysis)
        if collapsed:
            repairs.append(
                {
                    "field": "line1",
                    "position": "name",
                    "from": name_zone,
                    "to": repaired,
                    "reason": "collapse_given_name_filler_noise",
                }
            )
            name_zone = repaired
            analysis = analyze_td3_line1_given_prefix(name_zone)

        # If noise remains after the best prefix and there is no clean secondary token,
        # force the suffix to filler for a stricter TD3-like shape.
        if analysis["prefix"] and not analysis["clean_secondary_token"] and analysis["trailing_letters"] > 0:
            suffix_start = analysis["prefix_end"]
            forced = name_zone[:suffix_start] + re.sub(r"[A-Z0-9]", "<", name_zone[suffix_start:])
            repaired = forced
        else:
            repaired = name_zone

        if repaired != name_zone:
            repairs.append(
                {
                    "field": "line1",
                    "position": "name",
                    "from": name_zone,
                    "to": repaired,
                    "reason": "given_name_suffix_filler_enforce",
                }
            )
            name_zone = repaired

    name_zone = name_zone[:39].ljust(39, "<")
    line = "".join(chars[:5]) + name_zone
    line = _enforce_td3_length(line)
    return line, repairs


def repair_td3_line2(line2: str) -> tuple[str, list, dict]:
    line = normalize_td3_line2(line2)
    repairs = []
    chars = list(line)

    for i in range(10, 13):
        mapped = _to_letter(chars[i])
        if mapped != chars[i]:
            repairs.append(
                {
                    "field": "line2_nationality",
                    "position": i,
                    "from": chars[i],
                    "to": mapped,
                    "reason": "nationality_letter_fix",
                }
            )
            chars[i] = mapped

    if chars[20] not in ("M", "F", "<"):
        repairs.append(
            {"field": "line2", "position": 20, "from": chars[20], "to": "<", "reason": "normalize_sex"}
        )
        chars[20] = "<"

    line = "".join(chars)

    doc = _repair_checksum_field(
        "document_number",
        line[0:9],
        line[9],
        allowed_chars=MRZ_WHITELIST,
        max_changes=2,
    )
    dob = _repair_checksum_field(
        "birth_date",
        line[13:19],
        line[19],
        allowed_chars="0123456789",
        max_changes=2,
    )
    exp = _repair_checksum_field(
        "expiry_date",
        line[21:27],
        line[27],
        allowed_chars="0123456789",
        max_changes=2,
    )
    personal = _repair_checksum_field(
        "personal_number",
        line[28:42],
        line[42],
        allowed_chars=MRZ_WHITELIST,
        max_changes=3,
        allow_blank_check=True,
    )

    repairs.extend(doc["repairs"])
    repairs.extend(dob["repairs"])
    repairs.extend(exp["repairs"])
    repairs.extend(personal["repairs"])

    line = (
        doc["data"]
        + doc["check"]
        + line[10:13]
        + dob["data"]
        + dob["check"]
        + line[20]
        + exp["data"]
        + exp["check"]
        + personal["data"]
        + personal["check"]
        + line[43]
    )

    composite_data = line[0:10] + line[13:20] + line[21:43]
    expected_final = checksum(composite_data)
    if line[43] != expected_final:
        repairs.append(
            {
                "field": "final_check",
                "position": 43,
                "from": line[43],
                "to": expected_final,
                "reason": "composite_check_recomputed",
            }
        )
        line = line[:43] + expected_final

    line = normalize_td3_line2(line)
    checks = validate_td3_checks(line)
    return line, repairs, checks


def validate_and_correct_mrz(line1, line2):
    # Backward-compatible wrapper around the deeper repair pipeline.
    fixed_line1, _ = repair_td3_line1(line1)
    fixed_line2, _, _ = repair_td3_line2(line2)
    return fixed_line1, fixed_line2


def parse_mrz_fields(line1: str, line2: str) -> dict:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    names_raw = line1[5:44]
    name_parts = names_raw.split("<<", 1)
    surname = (name_parts[0] if name_parts else "").replace("<", " ").strip()
    given = (name_parts[1] if len(name_parts) > 1 else "").replace("<", " ").strip()

    return {
        "document_type": line1[0:2].replace("<", ""),
        "issuing_country": line1[2:5].replace("<", ""),
        "surname": surname,
        "given_names": given,
        "document_number": line2[0:9].replace("<", ""),
        "document_number_check": line2[9],
        "nationality": line2[10:13].replace("<", ""),
        "birth_date_yymmdd": line2[13:19],
        "birth_date_check": line2[19],
        "sex": line2[20].replace("<", ""),
        "expiry_date_yymmdd": line2[21:27],
        "expiry_date_check": line2[27],
        "personal_number": line2[28:42].replace("<", ""),
        "personal_number_check": line2[42],
        "final_check": line2[43],
    }

# -------------------------------------------------------
# CANDIDATE GENERATION
# -------------------------------------------------------

def _ocr_variant(variant_img: np.ndarray, psm: int) -> str:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={MRZ_WHITELIST}"
    text = pytesseract.image_to_string(variant_img, config=cfg)
    return normalize_mrz(text)


def generate_ocr_candidates(line_img: np.ndarray, line_index: int) -> list[dict]:
    line_name = f"line{line_index}"
    variants = _line_variants(line_img, line_name)

    best_by_text = {}
    for variant in variants:
        for psm in PSM_MODES:
            raw = _ocr_variant(variant["image"], psm=psm)
            if line_index == 1:
                normalized = normalize_td3_line1(raw)
                score, breakdown = score_td3_line1(normalized)
                check_pass = None
            else:
                normalized = normalize_td3_line2(raw)
                score, breakdown = score_td3_line2(normalized)
                check_pass = breakdown.get("checks_passed")

            existing = best_by_text.get(normalized)
            candidate = {
                "line_index": line_index,
                "text": normalized,
                "raw_text": raw,
                "score": float(score),
                "score_breakdown": breakdown,
                "variant_id": variant["id"],
                "variant_meta": {
                    "scale": variant["scale"],
                    "source": variant["source"],
                    "threshold": variant["threshold"],
                    "morph": variant["morph"],
                },
                "psm": psm,
                "checksum_pass_count": check_pass,
                "variant_image": variant["image"],
            }
            if (existing is None) or (candidate["score"] > existing["score"]):
                best_by_text[normalized] = candidate

    candidates = sorted(best_by_text.values(), key=lambda x: x["score"], reverse=True)
    return candidates


def _pair_candidates(line1_candidates: list[dict], line2_candidates: list[dict]) -> dict:
    top_line1 = line1_candidates[:6]
    top_line2 = line2_candidates[:6]

    scored_pairs = []
    for c1 in top_line1:
        for c2 in top_line2:
            fallback_country = c2["text"][10:13]
            repaired1, repairs1 = repair_td3_line1(c1["text"], fallback_country=fallback_country)
            repaired2, repairs2, checks = repair_td3_line2(c2["text"])

            s1, d1 = score_td3_line1(repaired1)
            s2, d2 = score_td3_line2(repaired2)
            consistency_bonus = 0.0

            issuing = repaired1[2:5]
            nationality = repaired2[10:13]
            if re.fullmatch(r"[A-Z]{3}", issuing) and issuing == nationality:
                consistency_bonus += 3.0

            pair_score = s1 + s2 + checks["passed_count"] * 8 + consistency_bonus

            scored_pairs.append(
                {
                    "line1": repaired1,
                    "line2": repaired2,
                    "line1_source": c1,
                    "line2_source": c2,
                    "score": float(pair_score),
                    "line1_score": float(s1),
                    "line2_score": float(s2),
                    "line1_breakdown": d1,
                    "line2_breakdown": d2,
                    "checks": checks,
                    "repairs": repairs1 + repairs2,
                }
            )

    if not scored_pairs:
        raise RuntimeError("No OCR candidate pairs produced")

    scored_pairs.sort(key=lambda x: x["score"], reverse=True)
    best = scored_pairs[0]
    best["pairs_evaluated"] = len(scored_pairs)
    best["top_pairs"] = scored_pairs[:3]
    return best


def build_td3_report(best_pair: dict, split_meta: dict, line1_candidates: list[dict], line2_candidates: list[dict]) -> dict:
    parsed = parse_mrz_fields(best_pair["line1"], best_pair["line2"])
    checks = validate_td3_checks(best_pair["line2"])

    def _candidate_summary(cands: list[dict], top_n: int = 5) -> list[dict]:
        out = []
        for c in cands[:top_n]:
            out.append(
                {
                    "text": c["text"],
                    "score": c["score"],
                    "variant_id": c["variant_id"],
                    "psm": c["psm"],
                    "checksum_pass_count": c["checksum_pass_count"],
                }
            )
        return out

    return {
        "split": split_meta,
        "candidate_stats": {
            "line1_candidates": len(line1_candidates),
            "line2_candidates": len(line2_candidates),
            "pairs_evaluated": best_pair.get("pairs_evaluated", 0),
            "line1_top": _candidate_summary(line1_candidates),
            "line2_top": _candidate_summary(line2_candidates),
        },
        "selected": {
            "line1": {
                "score": best_pair["line1_score"],
                "text": best_pair["line1"],
                "variant_id": best_pair["line1_source"]["variant_id"],
                "psm": best_pair["line1_source"]["psm"],
                "variant_meta": best_pair["line1_source"]["variant_meta"],
            },
            "line2": {
                "score": best_pair["line2_score"],
                "text": best_pair["line2"],
                "variant_id": best_pair["line2_source"]["variant_id"],
                "psm": best_pair["line2_source"]["psm"],
                "variant_meta": best_pair["line2_source"]["variant_meta"],
            },
            "pair_score": best_pair["score"],
        },
        "checksum_summary": checks,
        "repairs_applied": best_pair["repairs"],
        "parsed_fields": parsed,
    }

# -------------------------------------------------------
# OCR A FULL MRZ LINE (compat helper)
# -------------------------------------------------------

def ocr_line(img):
    candidates = generate_ocr_candidates(img, line_index=1)
    if not candidates:
        return ""
    return candidates[0]["text"]


# -------------------------------------------------------
# OCR PIPELINE
# -------------------------------------------------------

def run_ocr(mrz_img):
    # Accept either path or image
    if isinstance(mrz_img, str):
        img = cv2.imread(mrz_img, cv2.IMREAD_GRAYSCALE)
    else:
        img = _ensure_gray(mrz_img)

    if img is None:
        raise RuntimeError("Cannot load MRZ image")

    _save_global_preprocess_debug(img)
    line1_img, line2_img, split_meta = split_mrz_lines(img)

    line1_candidates = generate_ocr_candidates(line1_img, line_index=1)
    line2_candidates = generate_ocr_candidates(line2_img, line_index=2)
    if not line1_candidates or not line2_candidates:
        raise RuntimeError("No OCR candidates generated for MRZ lines")

    best_pair = _pair_candidates(line1_candidates, line2_candidates)
    report = build_td3_report(best_pair, split_meta, line1_candidates, line2_candidates)

    # Save best preprocessing variants for reproducibility/debug.
    save(best_pair["line1_source"]["variant_image"], "best_variant_line1.png")
    save(best_pair["line2_source"]["variant_image"], "best_variant_line2.png")

    print("\n[OCR] Candidate winner")
    print("-" * 60)
    print(f"line1: {best_pair['line1']}")
    print(f"line2: {best_pair['line2']}")
    print(f"pair score: {best_pair['score']:.2f}")
    print(f"line1 variant: {best_pair['line1_source']['variant_id']}  psm={best_pair['line1_source']['psm']}")
    print(f"line2 variant: {best_pair['line2_source']['variant_id']}  psm={best_pair['line2_source']['psm']}")
    print(
        "checks passed: "
        f"{best_pair['checks']['passed_count']}/{best_pair['checks']['total_count']} "
        f"(composite={best_pair['checks']['composite_valid']})"
    )
    print(f"repairs applied: {len(best_pair['repairs'])}")
    for item in best_pair["repairs"][:10]:
        print(f"  - {item}")
    if len(best_pair["repairs"]) > 10:
        print(f"  - ... {len(best_pair['repairs']) - 10} more")
    print("-" * 60)

    print("\nFinal MRZ")
    print("--------------------------------------------")
    print(best_pair["line1"])
    print(best_pair["line2"])
    print("--------------------------------------------")

    return best_pair["line1"], best_pair["line2"], report


# -------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    mrz_image = os.path.join(OUTPUT_DIR, "mrz_region.png")
    run_ocr(mrz_image)
