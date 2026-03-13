import json
import os
from datetime import datetime, timezone


def _clean_mrz_name(value: str) -> str:
    return value.replace("<", " ").strip()


def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    return 0


def _checksum(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(data):
        total += _char_value(c) * weights[i % 3]
    return str(total % 10)


def _validate_td3_checks(line2: str) -> dict:
    line2 = line2.ljust(44, "<")[:44]

    document = line2[0:9]
    document_check = line2[9]
    dob = line2[13:19]
    dob_check = line2[19]
    expiry = line2[21:27]
    expiry_check = line2[27]
    personal = line2[28:42]
    personal_check = line2[42]
    final_check = line2[43]

    personal_expected = _checksum(personal)
    personal_valid = personal_check == "<" and personal == "<" * 14
    if not personal_valid:
        personal_valid = personal_check == personal_expected

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]

    checks = {
        "document": {
            "actual": document_check,
            "expected": _checksum(document),
        },
        "birth_date": {
            "actual": dob_check,
            "expected": _checksum(dob),
        },
        "expiry_date": {
            "actual": expiry_check,
            "expected": _checksum(expiry),
        },
        "personal_number": {
            "actual": personal_check,
            "expected": personal_expected,
            "valid": personal_valid,
        },
        "composite": {
            "actual": final_check,
            "expected": _checksum(composite_data),
        },
    }

    checks["document"]["valid"] = checks["document"]["actual"] == checks["document"]["expected"]
    checks["birth_date"]["valid"] = checks["birth_date"]["actual"] == checks["birth_date"]["expected"]
    checks["expiry_date"]["valid"] = checks["expiry_date"]["actual"] == checks["expiry_date"]["expected"]
    checks["composite"]["valid"] = checks["composite"]["actual"] == checks["composite"]["expected"]

    checks["passed_count"] = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("valid") is True)
    checks["total_count"] = 5
    return checks


def parse_mrz_td3(line1: str, line2: str) -> dict:
    """Parse basic fields from TD3-style passport MRZ lines."""
    if not line1 or not line2:
        return {}

    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]

    names_raw = line1[5:44]
    name_parts = names_raw.split("<<", 1)
    surname = _clean_mrz_name(name_parts[0]) if name_parts else ""
    given_names = _clean_mrz_name(name_parts[1]) if len(name_parts) > 1 else ""

    parsed = {
        "document_type": line1[0:2].replace("<", ""),
        "issuing_country": line1[2:5].replace("<", ""),
        "surname": surname,
        "given_names": given_names,
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
        "checks": _validate_td3_checks(line2),
    }
    return parsed


def write_pipeline_report(output_dir: str, report_data: dict, filename: str = "report.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    sample_name = os.path.basename(os.path.normpath(output_dir)) or "report"
    prefixed_filename = f"{sample_name}_{filename}" if not filename.startswith(f"{sample_name}_") else filename
    out_path = os.path.join(output_dir, prefixed_filename)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_file": os.path.basename(out_path),
        **report_data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
