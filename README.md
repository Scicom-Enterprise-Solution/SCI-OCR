# Passport MRZ OCR Pipeline

This project extracts Machine Readable Zone (MRZ) text from a passport PDF in 3 stages:

1. Preprocess passport page (render + perspective correction)
2. Detect and crop MRZ region
3. OCR and normalize MRZ text

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
- `PDF_PATH` - Default PDF path used when no CLI argument is provided
- `OUTPUT_DIR` - Output folder for generated images

For better MRZ accuracy, an OCR-B model can be added manually to the Tesseract tessdata directory. In this environment, `ocrb.traineddata` was installed and the local `.env` uses `ocrb`. For accuracy-first extraction, keep `FAST_OCR=False`.

Template:

- `.env.example` (tracked)
- `.env` (local, git-ignored)

## Run

```bash
python run_pipeline.py <path-to-passport.pdf-or-image>
```

For the current local setup, `FAST_OCR=False` is the default via `.env`, so a normal run uses the exhaustive OCR profile.

Examples:

```bash
python run_pipeline.py samples/sss.png
python run_pipeline.py samples/101359705.pdf
FAST_OCR=true python run_pipeline.py samples/xcxc.png
FAST_OCR=false python run_pipeline.py samples/xcxc.png
python run_pipeline.py samples/sadia.png
OCR_BACKEND=auto python run_pipeline.py samples/sadia.png
```

Use `FAST_OCR=true` only for quick smoke tests. For production MRZ extraction, keep the exhaustive profile enabled.

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

Recommended:

- `OCR_BACKEND=paddle` as the current default and fastest production path on Linux when Paddle is configured correctly
- `OCR_BACKEND=auto` to combine PaddleOCR and Tesseract candidates when you want a hybrid check
- `OCR_BACKEND=tesseract` for baseline or fallback comparisons

## Tests

Run unit tests:

```bash
python -m unittest -q
```

## Reference Samples

The `samples/` directory is tracked in git and contains the sample PDFs, PNGs, and MRZ reference data used for regression checks.

- `samples/reference_clean.json` - filename-keyed MRZ reference data for the current clean reference set

## Output

Pipeline artifacts are written to `output/<sample-name>/`.

Typical files include:

- `aligned_passport.png`
- `mrz_region.png`
- `mrz_detected.png`
- `mrz_clean.png`
- `<sample-name>_report.json`

## Backend Regression Scripts

Run the full referenced sample set with a specific OCR backend:

```bash
python scripts/check_reference_set_paddle.py
python scripts/check_reference_set_auto.py
python scripts/check_reference_set_tesseract.py
```

## Main Files

- `run_pipeline.py` - Runs all stages end-to-end
- `preprocess_passport.py` - PDF render, contour detection, perspective correction
- `detect_mrz.py` - MRZ band detection and crop
- `ocr_mrz.py` - MRZ OCR and normalization
- `requirements.txt` - Python dependencies

## Notes

- `samples/` is tracked in git. `output/` remains git-ignored.
- If MRZ is not detected, check the debug images in `output/`.
