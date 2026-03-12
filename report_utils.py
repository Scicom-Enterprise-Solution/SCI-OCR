import json
import os
from datetime import datetime, timezone


def _clean_mrz_name(value: str) -> str:
    return value.replace("<", " ").strip()


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

    return {
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
    }


def write_pipeline_report(output_dir: str, report_data: dict, filename: str = "report.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **report_data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
