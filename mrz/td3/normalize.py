import re


MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_LINE_LEN = 44


def normalize_mrz(text: str) -> str:
    text = (text or "").upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9<]", "", text)
    text = text.replace("1ND", "IND")
    return text


def normalize_td3_line1(text: str) -> str:
    text = normalize_mrz(text)

    if text.startswith("P") and len(text) >= 2:
        second = text[1]
        if second == "0":
            text = "PO" + text[2:]
    elif text.startswith("P"):
        text = "P<"
    elif text.startswith(("PO", "P0")):
        text = "PO" + text[2:]

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


COUNTRY_CODE_AMBIGUOUS_SUBS = {
    "0": ("O", "Q"),
    "1": ("I",),
    "2": ("Z",),
    "5": ("S",),
    "6": ("G",),
    "8": ("B",),
}


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


def _normalize_numeric_field(data: str) -> str:
    data = normalize_mrz(data)
    return "".join(NUMERIC_FIELD_SUBS.get(ch, ch) for ch in data)


def _ambiguous_doc_char_count(data: str) -> int:
    return sum(1 for ch in normalize_mrz(data) if ch in set(NUMERIC_FIELD_SUBS))
