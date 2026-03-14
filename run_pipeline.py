import os
import sys

from document_inputs import DocumentInputError
from env_utils import load_env_file

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
import preprocess_passport as stage1


def main() -> None:
    env_input_path = os.getenv("PDF_PATH", stage1.PDF_PATH)
    input_path = sys.argv[1] if len(sys.argv) > 1 else env_input_path
    try:
        report = process_document(input_path, use_face_hint=USE_FACE_HINT)
    except DocumentInputError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    if report.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
