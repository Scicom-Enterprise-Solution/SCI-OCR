#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.reference_utils import normalize_reference_samples

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


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
    returncode: int | None = None,
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
        if returncode is not None:
            print(f"[Runner] code={returncode}")
    print()


def main() -> int:
    root = ROOT
    refs_path = root / "samples" / "reference_td3_clean.json"

    with refs_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    refs = normalize_reference_samples(payload)

    passes = []
    fails = []

    env = {**os.environ, "OCR_BACKEND": "paddle"}

    for filename, expected in refs.items():
        sample = root / "samples" / filename
        sample_name = sample.stem
        print(f"RUN {filename}", flush=True)
        started_at = time.perf_counter()

        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", str(sample)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        report = root / "output" / sample_name / f"{sample_name}_report.json"
        if not report.exists():
            print_sample_result(
                filename,
                ok=False,
                elapsed_s=time.perf_counter() - started_at,
                reason="missing_report",
                returncode=proc.returncode,
            )
            fails.append({
                "filename": filename,
                "reason": "missing_report",
                "returncode": proc.returncode,
            })
            continue

        with report.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        text = payload.get("mrz", {}).get("text", {})
        actual1 = text.get("line1", "")
        actual2 = text.get("line2", "")
        expected1 = expected.get("line1", "")
        expected2 = expected.get("line2", "")

        ok = (
            proc.returncode == 0
            and actual1 == expected1
            and actual2 == expected2
        )

        if ok:
            print_sample_result(
                filename,
                ok=True,
                elapsed_s=time.perf_counter() - started_at,
                actual1=actual1,
                actual2=actual2,
            )
            passes.append(filename)
        else:
            print_sample_result(
                filename,
                ok=False,
                elapsed_s=time.perf_counter() - started_at,
                actual1=actual1,
                actual2=actual2,
                expected1=expected1,
                expected2=expected2,
                reason="mismatch",
                returncode=proc.returncode,
            )
            fails.append({
                "filename": filename,
                "reason": "mismatch",
                "returncode": proc.returncode,
                "actual1": actual1,
                "expected1": expected1,
                "actual2": actual2,
                "expected2": expected2,
            })

    print()
    print(f"{color_status('PASS', True)} {len(passes)}")
    print(f"{color_status('FAIL', False)} {len(fails)}")

    for name in passes:
        print(f"OK {name}")

    for item in fails:
        print(f"FILE {item['filename']}")
        print(f"REASON {item['reason']}")
        print(f"CODE {item['returncode']}")
        if item["reason"] == "mismatch":
            print(f"ACTUAL1 {item['actual1']}")
            print(f"EXPECT1 {item['expected1']}")
            print(f"ACTUAL2 {item['actual2']}")
            print(f"EXPECT2 {item['expected2']}")

    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
