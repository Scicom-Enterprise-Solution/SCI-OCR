#!/usr/bin/env python3

import io
import json
import os
import pathlib
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_inputs import DocumentInputError
from pipelines.mrz_pipeline import process_document
from path_utils import to_repo_relative
from samples.reference_utils import normalize_reference_samples
from logger_utils import is_debug_enabled


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
RUN_REPORTS_DIR = ROOT / "samples" / "reports"


def color_status(label: str, ok: bool) -> str:
    return f"{GREEN if ok else RED}{label}{RESET}"


def print_sample_result(
    filename: str,
    *,
    ok: bool,
    elapsed_s: float | None = None,
    actual1: str = "",
    actual2: str = "",
    expected1: str = "",
    expected2: str = "",
    reason: str = "",
    error: str = "",
) -> None:
    suffix = f" ({elapsed_s:.2f}s)" if elapsed_s is not None else ""
    print(f"RESULT {filename} {color_status('PASS' if ok else 'FAIL', ok)}{suffix}")
    if actual1 or actual2:
        print("Extracted MRZ text:")
        print(actual1)
        print(actual2)
    if ok:
        print(
            f"[Reference] {color_status('PASS', True)} "
            f"(line1={color_status('PASS', True)}, line2={color_status('PASS', True)})"
        )
    else:
        line1_ok = actual1 == expected1
        line2_ok = actual2 == expected2
        print(
            f"[Reference] {color_status('FAIL', False)} "
            f"(line1={color_status('PASS' if line1_ok else 'FAIL', line1_ok)}, "
            f"line2={color_status('PASS' if line2_ok else 'FAIL', line2_ok)})"
        )
        if not line1_ok and expected1:
            print(f"[Reference] Expected line1: {expected1}")
        if not line2_ok and expected2:
            print(f"[Reference] Expected line2: {expected2}")
        if reason:
            print(f"[Runner] reason={reason}")
        if error:
            print(f"[Runner] error={error}")
    print()


def _build_combined_report_path(backend: str) -> pathlib.Path:
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%d%m%y%H%M%S", time.localtime())
    path = RUN_REPORTS_DIR / f"{timestamp}_{backend}.json"
    suffix = 1
    while path.exists():
        path = RUN_REPORTS_DIR / f"{timestamp}_{backend}_{suffix}.json"
        suffix += 1
    return path


def run_reference_set(backend: str) -> int:
    run_started_at = time.perf_counter()
    refs_path = ROOT / "samples" / "reference_td3_clean.json"
    with refs_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    refs = normalize_reference_samples(payload)

    passes = []
    fails = []
    sample_timings: list[tuple[str, float, bool]] = []
    sample_results: list[dict] = []

    original_backend = os.environ.get("OCR_BACKEND")
    os.environ["OCR_BACKEND"] = backend

    use_face_hint = os.getenv("USE_FACE_HINT", "False").strip().lower() in {"1", "true", "yes", "on"}
    debug = is_debug_enabled()

    try:
        for filename, expected in refs.items():
            sample = ROOT / "samples" / filename
            print(f"RUN {filename}", flush=True)
            started_at = time.perf_counter()

            try:
                if debug:
                    report = process_document(
                        str(sample),
                        use_face_hint=use_face_hint,
                        emit_progress=True,
                    )
                else:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer), redirect_stderr(buffer):
                        report = process_document(
                            str(sample),
                            use_face_hint=use_face_hint,
                            emit_progress=False,
                        )
                error = ""
            except DocumentInputError as exc:
                report = {"status": "failed", "error": str(exc), "mrz": {"text": {"line1": "", "line2": ""}}}
                error = str(exc)
            except Exception as exc:
                report = {"status": "failed", "error": str(exc), "mrz": {"text": {"line1": "", "line2": ""}}}
                error = str(exc)

            text = report.get("mrz", {}).get("text", {})
            actual1 = text.get("line1", "")
            actual2 = text.get("line2", "")
            expected1 = expected.get("line1", "")
            expected2 = expected.get("line2", "")

            ok = (
                report.get("status") == "success"
                and actual1 == expected1
                and actual2 == expected2
            )

            if ok:
                elapsed_s = time.perf_counter() - started_at
                print_sample_result(
                    filename,
                    ok=True,
                    elapsed_s=elapsed_s,
                    actual1=actual1,
                    actual2=actual2,
                )
                passes.append(filename)
                sample_timings.append((filename, elapsed_s, True))
                sample_results.append({
                    "filename": filename,
                    "status": "PASS",
                    "elapsed_s": round(elapsed_s, 2),
                    "line1": actual1,
                    "line2": actual2,
                })
            else:
                elapsed_s = time.perf_counter() - started_at
                print_sample_result(
                    filename,
                    ok=False,
                    elapsed_s=elapsed_s,
                    actual1=actual1,
                    actual2=actual2,
                    expected1=expected1,
                    expected2=expected2,
                    reason="mismatch" if report.get("status") == "success" else "run_failed",
                    error=error or report.get("error", ""),
                )
                sample_timings.append((filename, elapsed_s, False))
                fail_item = {
                    "filename": filename,
                    "reason": "mismatch" if report.get("status") == "success" else "run_failed",
                    "actual1": actual1,
                    "expected1": expected1,
                    "actual2": actual2,
                    "expected2": expected2,
                    "error": error or report.get("error", ""),
                }
                fails.append(fail_item)
                sample_results.append({
                    "filename": filename,
                    "status": "FAIL",
                    "elapsed_s": round(elapsed_s, 2),
                    "line1": actual1,
                    "line2": actual2,
                    "expected_line1": expected1,
                    "expected_line2": expected2,
                    "reason": fail_item["reason"],
                    "error": fail_item["error"],
                })
    finally:
        if original_backend is None:
            os.environ.pop("OCR_BACKEND", None)
        else:
            os.environ["OCR_BACKEND"] = original_backend

    print()
    print(f"{color_status('PASS', True)} {len(passes)}")
    print(f"{color_status('FAIL', False)} {len(fails)}")
    print()

    total_elapsed_s = time.perf_counter() - run_started_at
    warmup_sample = sample_timings[0] if sample_timings else None
    steady_state_timings = sample_timings[1:] if len(sample_timings) > 1 else []
    steady_total_s = sum(item[1] for item in steady_state_timings)
    steady_count = len(steady_state_timings)
    steady_avg_s = (steady_total_s / steady_count) if steady_count else 0.0
    steady_pass_timings = [item[1] for item in steady_state_timings if item[2]]
    steady_fail_timings = [item[1] for item in steady_state_timings if not item[2]]

    print(f"TOTAL_RAW {total_elapsed_s:.2f}s")
    if warmup_sample is not None:
        print(f"WARMUP_SAMPLE {warmup_sample[0]} ({warmup_sample[1]:.2f}s)")
    print(f"TOTAL {steady_total_s:.2f}s")
    print(f"AVG_PER_SAMPLE {steady_avg_s:.2f}s")
    if steady_pass_timings:
        print(f"AVG_PASS {sum(steady_pass_timings) / len(steady_pass_timings):.2f}s")
    if steady_fail_timings:
        print(f"AVG_FAIL {sum(steady_fail_timings) / len(steady_fail_timings):.2f}s")

    for item in fails:
        print(f"FILE {item['filename']}")
        print(f"REASON {item['reason']}")
        if item["reason"] == "mismatch":
            print(f"ACTUAL1 {item['actual1']}")
            print(f"EXPECT1 {item['expected1']}")
            print(f"ACTUAL2 {item['actual2']}")
            print(f"EXPECT2 {item['expected2']}")
        if item["error"]:
            print(f"ERROR {item['error']}")

    report_path = _build_combined_report_path(backend)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": {
                    "pass_count": len(passes),
                    "fail_count": len(fails),
                    "total_raw_s": round(total_elapsed_s, 2),
                    "warmup_sample": {
                        "filename": warmup_sample[0],
                        "elapsed_s": round(warmup_sample[1], 2),
                    } if warmup_sample else None,
                    "total_s": round(steady_total_s, 2),
                    "avg_per_sample_s": round(steady_avg_s, 2),
                    "avg_pass_s": round(sum(steady_pass_timings) / len(steady_pass_timings), 2) if steady_pass_timings else None,
                    "avg_fail_s": round(sum(steady_fail_timings) / len(steady_fail_timings), 2) if steady_fail_timings else None,
                },
                "samples": sample_results,
            },
            f,
            indent=2,
        )
    print(f"RUN_REPORT {to_repo_relative(str(report_path))}")

    return 0 if not fails else 1
