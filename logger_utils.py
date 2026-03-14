import json
import os
from typing import Any
from datetime import datetime, timezone


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _parse_bool_env("DEBUG", False)
LOG_FORMAT = (os.getenv("LOG_FORMAT", "human").strip().lower() or "human")


def is_debug_enabled() -> bool:
    return DEBUG


def use_json_logs() -> bool:
    return LOG_FORMAT == "json"


def log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    if use_json_logs():
        payload = {
            "level": level,
            "event": event,
            **fields,
        }
        print(json.dumps(payload, ensure_ascii=True))
        return

    message = fields.pop("message", "")
    prefix = f"[{level.upper()}]"
    if event:
        prefix += f" {event}"
    if message:
        print(f"{prefix} {message}")
    elif fields:
        print(f"{prefix} {fields}")
    else:
        print(prefix)


def print_reference_summary(comparison: dict) -> None:
    if not comparison.get("reference_available"):
        if use_json_logs():
            log_event("reference", reference_available=False)
        else:
            print("[Reference] No reference entry for this sample.")
        return

    line1_match = comparison.get("line1_match", False)
    line2_match = comparison.get("line2_match", False)
    exact_match = comparison.get("exact_match", False)

    if use_json_logs():
        log_event(
            "reference",
            status="PASS" if exact_match else "FAIL",
            line1="PASS" if line1_match else "FAIL",
            line2="PASS" if line2_match else "FAIL",
            expected=comparison.get("expected"),
        )
        return

    print(
        "[Reference] "
        f"{'PASS' if exact_match else 'FAIL'} "
        f"(line1={'PASS' if line1_match else 'FAIL'}, "
        f"line2={'PASS' if line2_match else 'FAIL'})"
    )

    if not line1_match:
        print(f"[Reference] Expected line1: {comparison['expected']['line1']}")
    if not line2_match:
        print(f"[Reference] Expected line2: {comparison['expected']['line2']}")


def print_final_report(report: dict) -> None:
    if report.get("status") != "success":
        if use_json_logs():
            log_event(
                "mrz_result",
                level="error",
                status=report.get("status"),
                error=report.get("error"),
                sample_name=report.get("input", {}).get("sample_name"),
                filename=report.get("input", {}).get("filename"),
                timestamp_utc=report.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
                duration_ms=report.get("duration_ms"),
                report_path=report.get("report_path"),
            )
        return

    text = report.get("mrz", {}).get("text", {})
    line1 = text.get("line1", "")
    line2 = text.get("line2", "")
    comparison = report.get("reference_comparison", {})
    input_meta = report.get("input", {})
    ocr_meta = report.get("mrz", {}).get("ocr", {})
    selected = ocr_meta.get("selected", {})
    checksum_summary = ocr_meta.get("checksum_summary", {})

    backend = None
    line1_backend = selected.get("line1", {}).get("psm")
    line2_backend = selected.get("line2", {}).get("psm")
    if line1_backend == line2_backend:
        backend = line1_backend
    elif line1_backend or line2_backend:
        backend = {"line1": line1_backend, "line2": line2_backend}

    if use_json_logs():
        log_event(
            "mrz_result",
            status=report.get("status"),
            timestamp_utc=report.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
            sample_name=input_meta.get("sample_name"),
            filename=input_meta.get("filename"),
            source_type=input_meta.get("source_type"),
            backend=backend,
            line1=line1,
            line2=line2,
            duration_ms=report.get("duration_ms"),
            checksum_summary=checksum_summary,
            pair_score=selected.get("pair_score"),
            reference=comparison,
            report_path=report.get("report_path"),
        )
        return

    print("Extracted MRZ text:")
    print(line1)
    print(line2)
    print()
    print_reference_summary(comparison)
