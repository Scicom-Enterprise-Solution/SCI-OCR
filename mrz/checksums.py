import itertools
import re

from mrz.country_codes import is_valid_mrz_country_code
from mrz.normalize import (
    _ambiguous_doc_char_count,
    _generate_country_code_variants,
    _normalize_numeric_field,
    normalize_mrz,
    normalize_td3_line1,
    normalize_td3_line2,
)


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

AMBIGUOUS_DOC_CHARS = set({
    "O", "Q", "I", "L", "Z", "S", "B", "G",
})

DOC_NUMBER_AMBIGUOUS_SUBS = {
    **AMBIGUOUS_FIELD_SUBS,
    "Q": ("0", "O"),
}


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
    if len(line2) < 44:
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

    if len(line2) < 44:
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
