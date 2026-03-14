import os
import shutil

import numpy as np
import pytesseract


def resolve_tesseract_cmd() -> str | None:
    env_tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    candidates = []

    if env_tesseract_cmd:
        candidates.append(env_tesseract_cmd)

    candidates.append("tesseract")

    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        resolved = shutil.which(expanded)
        if resolved:
            return resolved

    if env_tesseract_cmd:
        print(f"[WARN] TESSERACT_CMD not found: {env_tesseract_cmd}")

    return None


def configure_tesseract_cmd() -> str | None:
    resolved = resolve_tesseract_cmd()
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
    return resolved


def legacy_oem_supported(lang: str) -> bool:
    probe = np.full((24, 96), 255, dtype=np.uint8)

    try:
        pytesseract.image_to_string(
            probe,
            lang=lang,
            config=(
                "--oem 0 --psm 10 "
                "-c tessedit_char_whitelist=A "
                "-c load_system_dawg=0 "
                "-c load_freq_dawg=0"
            ),
        )
        return True
    except pytesseract.TesseractError as exc:
        message = str(exc).lower()
        if "legacy" in message and "components are not present" in message:
            return False
        print(f"[WARN] Could not validate legacy OCR support for '{lang}': {exc}")
        return False


def resolve_oems(lang: str, requested_oems: list[int]) -> list[int]:
    resolved = []
    legacy_checked = False
    legacy_supported = False

    for oem in requested_oems:
        if oem not in (0, 1, 2, 3):
            print(f"[WARN] Ignoring unsupported TESSERACT_OEMS value: {oem}")
            continue

        if oem in (0, 2):
            if not legacy_checked:
                legacy_supported = legacy_oem_supported(lang)
                legacy_checked = True
            if not legacy_supported:
                print(
                    f"[WARN] Skipping OEM {oem} for '{lang}' because legacy components are not available"
                )
                continue

        if oem not in resolved:
            resolved.append(oem)

    if resolved:
        return resolved

    print("[WARN] No usable Tesseract OEM configured; falling back to OEM 1")
    return [1]


def build_ocr_configs(oems: list[int], psms: list[int]) -> list[dict]:
    return [
        {"oem": oem, "psm": psm, "cfg": f"--oem {oem} --psm {psm}"}
        for oem in oems
        for psm in psms
    ]


def ocr_image(img, cfg: str, lang: str, whitelist: str, normalize_mrz) -> str:
    text = pytesseract.image_to_string(
        img,
        lang=lang,
        config=(
            f"{cfg} "
            f"-c tessedit_char_whitelist={whitelist} "
            f"-c load_system_dawg=0 "
            f"-c load_freq_dawg=0"
        ),
    )
    return normalize_mrz(text)
