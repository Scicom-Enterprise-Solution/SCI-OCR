#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    root = ROOT
    refs_path = root / "samples" / "reference_clean.json"

    with refs_path.open("r", encoding="utf-8") as f:
        refs = json.load(f)

    passes = []
    fails = []

    env = {**os.environ, "OCR_BACKEND": "tesseract"}

    for filename, expected in refs.items():
        sample = root / "samples" / filename
        sample_name = sample.stem
        print(f"RUN {filename}", flush=True)

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
            passes.append(filename)
        else:
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
    print(f"PASS {len(passes)}")
    print(f"FAIL {len(fails)}")

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
