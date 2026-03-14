import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from document_inputs import DocumentInputError
from env_utils import load_env_file
from logger_utils import is_debug_enabled, print_final_report, use_json_logs

load_env_file()

USE_FACE_HINT = os.getenv("USE_FACE_HINT", "True").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Import both stages. Each module is self-contained; we call their functions
# individually rather than running them as scripts, so we can pass data
# directly between stages without touching disk mid-pipeline.
# ---------------------------------------------------------------------------

from pipelines.mrz_pipeline import build_stage2_attempts, process_document
from document_preparation import passport as stage1


def main() -> None:
    env_input_path = os.getenv("PDF_PATH", stage1.PDF_PATH)
    input_path = sys.argv[1] if len(sys.argv) > 1 else env_input_path
    debug = is_debug_enabled()
    try:
        if debug:
            report = process_document(input_path, use_face_hint=USE_FACE_HINT, emit_progress=True)
        else:
            if use_json_logs():
                print(
                    '{"level": "info", "event": "mrz_processing", '
                    f'"filename": "{os.path.basename(input_path)}", '
                    '"status": "started"}',
                    flush=True,
                )
            else:
                print(f"Processing {os.path.basename(input_path)}...", flush=True)
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                report = process_document(input_path, use_face_hint=USE_FACE_HINT, emit_progress=False)
    except DocumentInputError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception:
        if not debug:
            try:
                output = buffer.getvalue()
            except Exception:
                output = ""
            if output.strip():
                print(output, end="" if output.endswith("\n") else "\n")
        raise

    if debug:
        return

    if report.get("status") == "success":
        print_final_report(report)
    else:
        output = buffer.getvalue()
        if output.strip():
            print(output, end="" if output.endswith("\n") else "\n")

    if report.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
