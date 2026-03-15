# OCR Pipeline Foundation

This repository is focused on passport MRZ extraction.

The current production-ready workflow is TD3 passport MRZ extraction in 3 stages:

1. Preprocess passport page (render + perspective correction)
2. Detect and crop MRZ region
3. OCR and normalize TD3 MRZ text

The current domain scope is intentionally narrow:

- passport MRZ only
- TD3 only
- 2 lines
- 44 characters per line

The codebase is split so OCR engines, document input loaders, task-specific pipeline logic, and the API layer can evolve separately.

## Requirements

- Python 3.10+
- Tesseract OCR installed and available on `PATH`
- Linux example: `sudo dnf install -y tesseract tesseract-langpack-eng`
- Note: Oracle Linux's packaged `eng` data is LSTM-only here, so legacy OEM modes `0` and `2` are not available unless you install a legacy-capable traineddata file manually.
- Optional MRZ improvement: install `ocrb.traineddata` and set `TESSERACT_LANG=eng+ocrb` or `ocrb`

Install Python packages:

```bash
pip install -r requirements.txt
```

For this project, prefer using the project venv explicitly:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Environment Variables

The project loads values from `.env` automatically (if present).

- `TESSERACT_CMD` - Optional command name or full path to the Tesseract executable. If unset, the code uses `tesseract` from `PATH`.
- `TESSERACT_LANG` - Optional Tesseract language(s), for example `eng` or `eng+ocrb`
- `TESSERACT_OEMS` - Optional comma-separated OEM list, for example `1` or `0,1`. Legacy OEMs require legacy-capable traineddata.
- `OCR_BACKEND` - OCR backend selection: `tesseract`, `paddle`, or `auto`
- `PADDLEOCR_LANG` - Optional PaddleOCR language code when PaddleOCR is enabled
- `PADDLEOCR_USE_GPU` - Optional boolean for PaddleOCR GPU inference
- `PADDLE_PDX_CACHE_HOME` - Optional Paddle model/cache directory. Prefer a project-relative path such as `.paddlex` for portability across servers.
- `FAST_OCR` - Optional speed-only mode. When `true`, Stage 3 uses a much smaller OCR search space for faster runs, but it can reduce name-field accuracy.
- `PADDLE_PROFILE` - Paddle search profile: `exhaustive`, `balanced`, or `fast`
- `PDF_PATH` - Default PDF path used when no CLI argument is provided
- `OUTPUT_DIR` - Output folder for generated images
- `ALIGNED_MAX_DIM` - Maximum longest-side dimension for the aligned passport working image before Stage 2/3
- `API_HOST` - API bind host
- `API_PORT` - API bind port
- `API_STORAGE_DIR` - API storage root for uploads, previews, reports, and SQLite DB
- `API_DB_PATH` - SQLite DB path for the API layer

For better MRZ accuracy, an OCR-B model can be added manually to the Tesseract tessdata directory. In this environment, `ocrb.traineddata` was installed and the local `.env` uses `ocrb`. For accuracy-first extraction, keep `FAST_OCR=False`.

Template:

- `.env.example` (tracked)
- `.env` (local, git-ignored)

## Run

```bash
./.venv/bin/python run_pipeline.py <path-to-passport.pdf-or-image>
```

For the current local setup, `FAST_OCR=False`, `PADDLE_PROFILE=exhaustive`, and `ALIGNED_MAX_DIM=1200` are the current accuracy-first defaults.

Examples:

```bash
./.venv/bin/python run_pipeline.py samples/sss.png
./.venv/bin/python run_pipeline.py samples/101359705.pdf
FAST_OCR=true ./.venv/bin/python run_pipeline.py samples/xcxc.png
FAST_OCR=false ./.venv/bin/python run_pipeline.py samples/xcxc.png
./.venv/bin/python run_pipeline.py samples/sadia.png
OCR_BACKEND=auto ./.venv/bin/python run_pipeline.py samples/sadia.png
DEBUG=True ./.venv/bin/python run_pipeline.py samples/sadia.png
LOG_FORMAT=json ./.venv/bin/python run_pipeline.py samples/sadia.png
```

Use `FAST_OCR=true` only for quick smoke tests. For production MRZ extraction, keep the exhaustive profile enabled.

Large-image note:

- very large source images can make MRZ OCR worse, not better
- after perspective correction, the pipeline now caps the aligned working image before Stage 2/3
- `ALIGNED_MAX_DIM=1200` is the current practical default
- for MRZ, a working range around `1000-1200` px on the long side has been more reliable than pushing oversized aligned images through OCR unchanged

## API Server

The repository includes a FastAPI server and SQLite-backed storage layer for:

- uploads
- preview generation
- extraction
- reviewer corrections / reference truth

Run the API DB initializer:

```bash
./.venv/bin/python scripts/init_db.py
```

Run the API server:

```bash
./.venv/bin/python scripts/run_api_dev.py
```

Default API docs:

- [http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs)

Current API routes:

- `GET /api/health`
- `POST /api/uploads`
- `POST /api/extractions`
- `GET /api/references`
- `POST /api/references`
- `GET /storage/...`

Upload response now includes:

- `document_id`
- `file_hash`
- `deduplicated`
- `preview_path`

Extraction request only requires:

```json
{
  "document_id": "..."
}
```

Optional fields:

- `crop`
- `rotation`
- `use_face_hint`

Default extraction behavior:

- original source image/page
- no crop
- no rotation
- no face detection unless explicitly requested

Notes:

- Uploads are deduplicated by SHA-256 hash.
- The API is intended to run as a long-lived process, so Paddle stays warm across requests.
- API report files are written under `storage/reports/`.
- API paths returned in JSON are repo-relative for portability.

## Storage And Reports

Runtime storage is kept outside git under `storage/`:

- `storage/uploads/`
- `storage/previews/`
- `storage/reports/`
- `storage/mrz.sqlite3`

CLI regression reports are separate from API reports:

- per-sample pipeline reports stay under `output/<sample>/`
- combined regression-run reports are written to `samples/reports/`

The regression runner writes one combined JSON report per run using a timestamped filename and prints the final path as `RUN_REPORT ...`.

Serialized paths in API responses and report JSON are stored as repo-relative paths with forward slashes for better portability across Linux, WSL, and Windows.

## Reference Regression Runs

Use the warm-process regression scripts for realistic Paddle performance:

```bash
./.venv/bin/python scripts/check_reference_set_paddle.py
./.venv/bin/python scripts/check_reference_set_tesseract.py
./.venv/bin/python scripts/check_reference_set_auto.py
```

These runners:

- keep the OCR backend warm across samples
- print per-sample MRZ output and reference verdict
- suppress internal stage spam when `DEBUG=False`
- write one combined run report to `samples/reports/`

Run summary fields:

- `TOTAL_RAW` - full run time including warm-up
- `WARMUP_SAMPLE` - first sample cold-start cost
- `TOTAL` - steady-state total excluding the first sample
- `AVG_PER_SAMPLE` - steady-state average excluding the first sample
- `AVG_PASS` / `AVG_FAIL` - steady-state pass/fail averages

## Architecture

The project is organized so reusable OCR infrastructure stays separate from MRZ-specific logic.

- `api/` - FastAPI server, routes, schemas, and service layer
- `db/` - SQLite persistence layer for uploads, extractions, and corrections
- `document_inputs/` - image and PDF loading/decoding
- `ocr_backends/` - OCR engine integrations such as PaddleOCR and Tesseract
- `pipelines/` - task orchestration layers
- `mrz/` - MRZ-specific packages organized by ICAO document format
- `mrz/td3/` - TD3 passport MRZ detection, normalization, checksums, repair, scoring, and OCR helpers
- `mrz/td3/ocr_pipeline.py` - TD3 passport OCR candidate scoring, repair, and normalization
- `report_utils.py` - per-run report writing and MRZ parsing helpers
- `run_pipeline.py` - CLI wrapper for local-file execution

The current end-to-end pipeline is MRZ-focused, but the surrounding structure is being shaped for broader document OCR workflows.

## PaddleOCR Backend

The project can run Stage 3 with PaddleOCR in addition to Tesseract. `OCR_BACKEND=paddle` is the current default.

### Linux GPU install

Use this when the machine has an NVIDIA GPU available to Linux and Paddle can see a CUDA device.

Important CUDA version note:

- the `CUDA Version` shown by `nvidia-smi` is the driver/runtime capability, not the exact Paddle wheel selector
- this project currently uses the Paddle `cu129` wheel on Linux GPU installs
- so a machine can show `CUDA Version: 13.0` in `nvidia-smi` and still correctly use the `cu129` Paddle wheel
- the Python environment must also contain the CUDA runtime packages expected by the installed Paddle GPU wheel
- if the host shows a newer CUDA version than the wheel target, that is usually fine as long as the Paddle wheel and its Python-side runtime packages match each other
- if they do not match, uninstall the current Paddle packages and reinstall the wheel set for the target runtime used by the project

```bash
.venv/bin/python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
.venv/bin/python -m pip install -r requirements-paddle-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

If you want to inspect what the Python environment currently has installed:

```bash
.venv/bin/python -m pip list | grep -E 'paddle|nvidia-'
```

Verify that the installed Paddle build can see the Linux GPU:

```bash
.venv/bin/python - <<'PY'
import paddle
print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.cuda.device_count())
PY
```

Expected result: CUDA support is `True` and the device count is at least `1`.

Then use:

```bash
export OCR_BACKEND=paddle
export PADDLEOCR_USE_GPU=True
```

### Linux CPU install

Use this on Linux systems without a GPU, or when you explicitly want PaddleOCR on CPU.

```bash
.venv/bin/python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
.venv/bin/python -m pip install -r requirements-paddle-cpu.txt
```

Then use:

```bash
export OCR_BACKEND=paddle
export PADDLEOCR_USE_GPU=False
```

Then run with one of:

```bash
.venv/bin/python run_pipeline.py samples/sadia.png
OCR_BACKEND=auto .venv/bin/python run_pipeline.py samples/sadia.png
OCR_BACKEND=tesseract .venv/bin/python run_pipeline.py samples/sadia.png
```

Default CLI output is concise: a short `Processing ...` line, the final extracted MRZ, and the reference result. Set `DEBUG=True` to restore verbose stage-by-stage logs. Set `LOG_FORMAT=json` to emit structured result events suitable for log ingestion.

Recommended:

- `OCR_BACKEND=paddle` as the current default and fastest production path on Linux when Paddle is configured correctly
- `OCR_BACKEND=auto` to combine PaddleOCR and Tesseract candidates when you want a hybrid check
- `OCR_BACKEND=tesseract` for baseline or fallback comparisons

## Tests

Run unit tests:

```bash
./.venv/bin/python -m unittest -q
```

Run focused API/database checks:

```bash
./.venv/bin/python -m unittest -q test_api_services.py test_db_sqlite.py test_report_utils.py
```

## Reference Samples

The `samples/` directory is tracked in git and contains the sample PDFs, PNGs, and MRZ reference data used for regression checks.

- `samples/reference_td3_clean.json` - TD3 passport reference set with metadata and a `samples` array of `{filename, line1, line2}` objects for the current clean visual-truth benchmark

## Output

Pipeline artifacts are written to `output/<sample-name>/`.

Typical files include:

- `aligned_passport.png`
- `mrz_region.png`
- `mrz_detected.png`
- `mrz_clean.png`
- `<sample-name>_report.json`

API runtime files are written under `storage/`:

- `storage/uploads/`
- `storage/previews/`
- `storage/reports/`
- `storage/mrz.sqlite3`

These are git-ignored.

## Backend Regression Scripts

Run the full referenced sample set with a specific OCR backend:

```bash
./.venv/bin/python scripts/check_reference_set_paddle.py
./.venv/bin/python scripts/check_reference_set_auto.py
./.venv/bin/python scripts/check_reference_set_tesseract.py
```

These scripts now run in-process, not one subprocess per sample. That means:

- OCR runtimes stay warm across the whole batch
- GPU utilization reflects a real long-lived service more closely
- end-of-run timing summary is more meaningful

Each regression run writes one combined report JSON to:

- `samples/reports/`

The combined run report includes:

- pass/fail counts
- warm-up sample timing
- steady-state totals
- average per-sample timing
- per-sample result details

## Main Files

- `run_pipeline.py` - CLI entrypoint for local-file testing
- `api/app.py` - FastAPI entrypoint
- `pipelines/mrz_pipeline.py` - reusable MRZ service pipeline
- `db/sqlite.py` - SQLite schema and persistence helpers
- `document_inputs/` - document file loaders for images and PDFs
- `ocr_backends/` - OCR backend integrations
- `document_preparation/passport.py` - contour detection and perspective correction
- `mrz/td3/detect.py` - TD3 passport MRZ band detection and crop
- `mrz/td3/ocr_pipeline.py` - TD3 passport MRZ OCR scoring and normalization
- `requirements.txt` - Python dependencies

## Notes

- `samples/` is tracked in git.
- `output/`, `storage/`, and `samples/reports/` are git-ignored runtime/generated paths.
- Stored/returned file paths are normalized to repo-relative form for portability across machines and OSes.
- If MRZ is not detected, check the debug images in `output/`.
