import re

from mrz.td3.country_codes import is_valid_mrz_country_code
from mrz.td3.checksums import validate_td3_checks
from mrz.td3.normalize import (
    MRZ_LINE_LEN,
    _ambiguous_doc_char_count,
    _sanitize_name_token,
    _split_tail_name_tokens,
    normalize_td3_line1,
    normalize_td3_line2,
)

PAIR_COUNTRY_MATCH_BONUS = 10.0

KNOWN_TD3_PASSPORT_DOCUMENT_CODES = {
    "P<", "PP", "PB", "PD", "PE", "PL", "PM",
    "PO", "PR", "PS", "PT", "PU",
}

# ------------------------------
# Existing scoring functions (unchanged)
# ------------------------------

def _max_consonant_run(token: str) -> int:
    vowels = set("AEIOUY")
    best = cur = 0
    for ch in token:
        if ch in vowels:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def _max_vowel_run(token: str) -> int:
    vowels = set("AEIOUY")
    best = cur = 0
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

    if 3 <= n <= 12:
        score += 12
    elif 2 <= n <= 14:
        score += 6
    else:
        score -= abs(n - 8) * 1.5

    if 9 <= n <= 12:
        score += 8

    if vowels >= 2:
        score += 10
    elif vowels == 1:
        score += 2
    else:
        score -= 12

    if token[-1] in "AEIOUY":
        score += 5

    if n == 3 and is_valid_mrz_country_code(token):
        score -= 8

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

    document_code = text[:2]
    if not (len(text) >= 2 and text[0] == "P" and re.fullmatch(r"[A-Z<]", text[1])):
        score -= 20
    elif document_code == "P<":
        score += 20
    elif document_code in KNOWN_TD3_PASSPORT_DOCUMENT_CODES:
        score += 16
    else:
        score += 4

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
        surname_raw, given_raw = name_zone.split("<<", 1)
        surname = _sanitize_name_token(surname_raw)
        first_token_raw = given_raw.split("<", 1)[0]
        first_token = _sanitize_name_token(first_token_raw)
        filler_tail = given_raw[len(first_token_raw):]

        if surname:
            score += 8
        else:
            score -= 10

        if first_token:
            score += 8
            token_score = _name_token_score(first_token)
            score += max(-8, min(24, token_score))
            score += max(-4, min(6, token_score / 4.0))
        else:
            score -= 6.0
            filler_tail = given_raw

        tail_tokens, tail_remainder = _split_tail_name_tokens(filler_tail)
        for tail_token in tail_tokens:
            token_score = _name_token_score(tail_token)
            score += 6
            score += max(-6, min(14, token_score))
            if 2 <= len(tail_token) <= 12:
                score += 4

        short_tail_tokens = [token for token in tail_tokens if 1 <= len(token) <= 5]
        if len(short_tail_tokens) >= 2 and filler_tail.count("<") >= 8:
            score -= len(short_tail_tokens) * 10.0

        suspicious_tail_fragments = sum(
            1
            for token in short_tail_tokens
            if len(set(token)) <= 3 or _name_token_score(token) <= 0
        )
        if suspicious_tail_fragments >= 2:
            score -= suspicious_tail_fragments * 12.0

        if re.search(r"<{8,}[A-Z]{1,4}<{2,}$", filler_tail):
            score -= 24.0

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

    if len(text) == MRZ_LINE_LEN:
        score += 20

    checks = validate_td3_checks(text)
    score += checks["passed_count"] * 18

    return score, checks


def pair_consistency_bonus(line1: str, line2: str) -> float:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    issuing = line1[2:5]
    nationality = line2[10:13]

    if (
        is_valid_mrz_country_code(issuing)
        and is_valid_mrz_country_code(nationality)
        and issuing == nationality
    ):
        return PAIR_COUNTRY_MATCH_BONUS

    return 0.0


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


# ------------------------------
# Utilities
# ------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(float(v), 1.0))


def _normalize_margin(raw_margin: float | None, low=2.5, high=12.0) -> float:
    if raw_margin is None:
        return 0.5
    if raw_margin <= low:
        return 0.0
    if raw_margin >= high:
        return 1.0
    return (raw_margin - low) / (high - low)


def _normalize_support(support_count: int | None, full=3) -> float:
    support_count = int(support_count or 0)
    if support_count <= 1:
        return 0.0
    return _clamp01((support_count - 1) / max(1, full - 1))


# ------------------------------
# STRUCTURE CONFIDENCE
# ------------------------------

def build_structure_confidence(line1: str, line2: str) -> dict:
    line1 = normalize_td3_line1(line1)
    line2 = normalize_td3_line2(line2)

    score = 1.0 if len(line1) == 44 and len(line2) == 44 else 0.7

    return {
        "score": score,
        "warnings": [] if score > 0.9 else ["structure_incomplete"]
    }


# ------------------------------
# MARGIN CONFIDENCE (FIXED)
# ------------------------------

def build_candidate_margin_confidence(best_pair, line1_ranked, line2_ranked):

    def compute(ranked, selected):
        selected_text = selected["text"]
        selected_score = selected["score"]

        same = [c for c in ranked if c["text"] == selected_text]
        support = len(same)

        support_score = _normalize_support(support)

        best_other = None
        for c in ranked:
            if c["text"] == selected_text:
                continue
            s = c["score"]
            if best_other is None or s > best_other:
                best_other = s

        if best_other is None:
            return {
                "score": max(0.9, support_score),
                "support": support,
                "support_score": support_score,
                "margin": None,
                "runner": False
            }

        margin = selected_score - best_other
        margin_score = _normalize_margin(margin)

        score = (margin_score * 0.7) + (support_score * 0.3)

        return {
            "score": _clamp01(score),
            "support": support,
            "support_score": support_score,
            "margin": margin,
            "runner": True
        }

    l1 = compute(line1_ranked, best_pair["line1"])
    l2 = compute(line2_ranked, best_pair["line2"])

    aggregate = (l1["score"] * 0.5) + (l2["score"] * 0.5)

    warnings = []

    if (
        aggregate < 0.25
        or (l1["runner"] and l1["score"] < 0.30 and l1["support"] < 2)
        or (l2["runner"] and l2["score"] < 0.30 and l2["support"] < 2)
    ):
        warnings.append("selected_candidate_margin_small")

    return {
        "score": _clamp01(aggregate),
        "line1_margin": round(l1["margin"], 4) if l1["margin"] is not None else None,
        "line2_margin": round(l2["margin"], 4) if l2["margin"] is not None else None,
        "pair_margin": None,
        "line1_margin_score": round(l1["score"], 4),
        "line2_margin_score": round(l2["score"], 4),
        "pair_margin_score": None,
        "line1_runner_up_available": l1["runner"],
        "line2_runner_up_available": l2["runner"],
        "pair_runner_up_available": False,
        "line1_same_text_support": l1["support"],
        "line2_same_text_support": l2["support"],
        "pair_same_text_support": min(l1["support"], l2["support"]),
        "warnings": warnings
    }


# ------------------------------
# FINAL CONFIDENCE (FIXED)
# ------------------------------

def assemble_td3_confidence(
    *,
    checksum_confidence,
    structure_confidence,
    repair_confidence,
    margin_confidence,
    ocr_confidence=None,
    selected_meta=None,
):

    base_weights = {
        "checksum": 0.4,
        "structure": 0.15,
        "repair": 0.15,
        "margin": 0.2,
        "ocr": 0.1,
    }

    ocr_available = bool(ocr_confidence and ocr_confidence.get("available") and ocr_confidence.get("score") is not None)

    weights = dict(base_weights)
    if not ocr_available:
        weights["ocr"] = 0.0

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    score = (
        weights["checksum"] * checksum_confidence["score"]
        + weights["structure"] * structure_confidence["score"]
        + weights["repair"] * repair_confidence["score"]
        + weights["margin"] * margin_confidence["score"]
    )

    if ocr_available:
        score += weights["ocr"] * ocr_confidence["score"]

    warnings = []
    for w in (
        checksum_confidence,
        structure_confidence,
        repair_confidence,
        margin_confidence,
    ):
        warnings += w.get("warnings", [])

    if not ocr_available:
        warnings.append("ocr_confidence_unavailable")

    if repair_confidence.get("repair_count", 0) > 0 and not checksum_confidence.get("composite_valid", False):
        warnings.append("repair_checksum_tension")

    # ---------------- CAP SYSTEM ----------------

    cap = 1.0

    if "composite_checksum_failed" in warnings or "repair_checksum_tension" in warnings:
        cap = min(cap, 0.65)

    if "line1_name_rewritten" in warnings or "multiple_repairs_applied" in warnings:
        cap = min(cap, 0.75)

    if "selected_candidate_margin_small" in warnings:
        cap = min(cap, 0.85)

    raw_score = _clamp01(score)
    score = min(raw_score, cap)

    # ---------------- SUSPICIOUS FIX ----------------

    suspicious = score < 0.6 or cap < 1.0

    return {
        "final_score": round(score, 4),
        "suspicious": suspicious,
        "warnings": list(dict.fromkeys(warnings)),
        "effective_weights": {
            "checksum": round(weights["checksum"], 4),
            "structure": round(weights["structure"], 4),
            "repair": round(weights["repair"], 4),
            "margin": round(weights["margin"], 4),
            "ocr_quality": round(weights["ocr"], 4),
        },
        "details": {
            "raw_final_score": round(raw_score, 4),
            "score_cap": round(cap, 4),
            "ocr_confidence_effective_score": round(ocr_confidence["score"], 4) if ocr_available else None,
            "selected_line1_score": round(float((selected_meta or {}).get("line1_score", 0.0)), 4),
            "selected_line2_score": round(float((selected_meta or {}).get("line2_score", 0.0)), 4),
            "selected_pair_score": round(float((selected_meta or {}).get("pair_score", 0.0)), 4),
        },
    }
