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
    "P<",
    "PP",
    "PB",
    "PD",
    "PE",
    "PL",
    "PM",
    "PO",
    "PR",
    "PS",
    "PT",
    "PU",
}


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
