# Passport MRZ OCR Pipeline

This project extracts Machine Readable Zone (MRZ) text from a passport PDF in 3 stages:

1. Preprocess passport page (render + perspective correction)
2. Detect and crop MRZ region
3. OCR and normalize MRZ text

## Requirements

- Python 3.10+
- Tesseract OCR installed on Windows:
  - C:\Program Files\Tesseract-OCR\tesseract.exe
  - or C:\Program Files (x86)\Tesseract-OCR\tesseract.exe

Install Python packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project loads values from `.env` automatically (if present).

- `TESSERACT_CMD` - Full path to `tesseract.exe`
- `PDF_PATH` - Default PDF path used when no CLI argument is provided
- `OUTPUT_DIR` - Output folder for generated images

Template:

- `.env.example` (tracked)
- `.env` (local, git-ignored)

## Run

```bash
python run_pipeline.py <path-to-passport.pdf>
```

Example:

```bash
python run_pipeline.py samples/passport.pdf
```

## Tests

Run unit tests:

```bash
python -m unittest -q
```

## Output

Pipeline artifacts are written to `output/` (for example: `aligned_passport.png`, `mrz_region.png`, `mrz_detected.png`, `mrz_clean.png`).

## Main Files

- `run_pipeline.py` - Runs all stages end-to-end
- `preprocess_passport.py` - PDF render, contour detection, perspective correction
- `detect_mrz.py` - MRZ band detection and crop
- `ocr_mrz.py` - MRZ OCR and normalization
- `requirements.txt` - Python dependencies

## Notes

- `samples/` and `output/` are ignored by Git in `.gitignore`.
- If MRZ is not detected, check the debug images in `output/`.
