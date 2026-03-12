# Passport MRZ OCR Pipeline

This project extracts Machine Readable Zone (MRZ) text from a passport PDF in 3 stages:

1. Preprocess passport page (render + perspective correction)
2. Detect and crop MRZ region
3. OCR and normalize MRZ text

## Requirements

- Python 3.10+
- Tesseract OCR installed and available on `PATH`
- Oracle Linux 9 example: `sudo dnf install -y tesseract tesseract-langpack-eng`
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
- `PDF_PATH` - Default PDF path used when no CLI argument is provided
- `OUTPUT_DIR` - Output folder for generated images

For better MRZ accuracy, an OCR-B model can be added manually to the Tesseract tessdata directory. In this environment, `ocrb.traineddata` was installed and the local `.env` uses `eng+ocrb`.

Template:

- `.env.example` (tracked)
- `.env` (local, git-ignored)

## Run

```bash
python run_pipeline.py <path-to-passport.pdf-or-image>
```

Examples:

```bash
python run_pipeline.py samples/sss.png
python run_pipeline.py samples/101359705.pdf
```

## Tests

Run unit tests:

```bash
python -m unittest -q
```

## Reference Samples

The `samples/` directory is tracked in git and contains the sample PDFs, PNGs, and MRZ reference data used for regression checks.

- `samples/mrz_reference_samples.json` - corrected MRZ reference data keyed by legacy numeric `sample_id`
- `samples/mrz_reference_samples_by_filename.json` - the same reference data keyed by filename (for example `sss.png`)

## Output

Pipeline artifacts are written to `output/<sample-name>/`.

Typical files include:

- `aligned_passport.png`
- `mrz_region.png`
- `mrz_detected.png`
- `mrz_clean.png`
- `<sample-name>_report.json`
- `report.json`

## Main Files

- `run_pipeline.py` - Runs all stages end-to-end
- `preprocess_passport.py` - PDF render, contour detection, perspective correction
- `detect_mrz.py` - MRZ band detection and crop
- `ocr_mrz.py` - MRZ OCR and normalization
- `requirements.txt` - Python dependencies

## Notes

- `samples/` is tracked in git. `output/` remains git-ignored.
- If MRZ is not detected, check the debug images in `output/`.
