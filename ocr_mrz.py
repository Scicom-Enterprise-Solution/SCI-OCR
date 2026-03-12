import os
import cv2
import pytesseract
import numpy as np
import re
from env_utils import load_env_file


load_env_file()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# -------------------------------------------------------
# TESSERACT PATH (Windows)
# -------------------------------------------------------

paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

env_tesseract_cmd = os.getenv("TESSERACT_CMD")

if env_tesseract_cmd and os.path.isfile(env_tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = env_tesseract_cmd
elif env_tesseract_cmd:
    print(f"[WARN] TESSERACT_CMD is set but file not found: {env_tesseract_cmd}")

if not getattr(pytesseract.pytesseract, "tesseract_cmd", "") or not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
    for p in paths:
        if os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


# -------------------------------------------------------
# NORMALIZE MRZ TEXT
# -------------------------------------------------------

def normalize_mrz(text):

    text = text.upper()
    text = text.replace(" ", "")

    text = re.sub(r"[^A-Z0-9<]", "", text)

    # Fix common nationality OCR error
    text = text.replace("1ND", "IND")

    # Remove stray letters before filler blocks
    text = re.sub(r"[A-Z](?=<{5,})", "", text)

    return text


# -------------------------------------------------------
# OCR A FULL MRZ LINE
# -------------------------------------------------------

def ocr_line(img):

    configs = [
        "--oem 1 --psm 7",
        "--oem 1 --psm 6",
        "--oem 1 --psm 13",
    ]

    best = ""

    for cfg in configs:

        text = pytesseract.image_to_string(
            img,
            config=f"{cfg} -c tessedit_char_whitelist={MRZ_WHITELIST}"
        )

        text = normalize_mrz(text)

        if len(text) > len(best):
            best = text

    return best


# -------------------------------------------------------
# ICAO CHECKSUM
# -------------------------------------------------------

def char_value(c):

    if c.isdigit():
        return int(c)

    if "A" <= c <= "Z":
        return ord(c) - 55

    return 0


def checksum(data):

    weights = [7,3,1]
    total = 0

    for i,c in enumerate(data):
        total += char_value(c) * weights[i % 3]

    return str(total % 10)


# -------------------------------------------------------
# FIELD CORRECTION
# -------------------------------------------------------

def correct_field(data, check):

    if checksum(data) == check:
        return data

    replacements = {
        "O":"0",
        "0":"O",
        "I":"1",
        "1":"I",
        "Z":"2",
        "2":"Z",
        "S":"5",
        "5":"S",
        "B":"8",
        "8":"B"
    }

    for i,c in enumerate(data):

        if c in replacements:

            candidate = data[:i] + replacements[c] + data[i+1:]

            if checksum(candidate) == check:
                return candidate

    return data


# -------------------------------------------------------
# VALIDATE MRZ
# -------------------------------------------------------

def validate_and_correct_mrz(line1, line2):

    if len(line2) < 44:
        return line1,line2

    doc_number = line2[0:9]
    doc_check = line2[9]

    dob = line2[13:19]
    dob_check = line2[19]

    expiry = line2[21:27]
    expiry_check = line2[27]

    personal = line2[28:42]
    personal_check = line2[42]

    doc_number = correct_field(doc_number, doc_check)
    dob = correct_field(dob, dob_check)
    expiry = correct_field(expiry, expiry_check)

    line2 = (
        doc_number
        + doc_check
        + line2[10:13]
        + dob
        + dob_check
        + line2[20]
        + expiry
        + expiry_check
        + personal
        + personal_check
        + line2[43]
    )

    return line1,line2


# -------------------------------------------------------
# CLEAN MRZ IMAGE (USED BY PIPELINE)
# -------------------------------------------------------

def clean_mrz_image(mrz_img):

    # Ensure grayscale
    if len(mrz_img.shape) == 3:
        img = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
    else:
        img = mrz_img.copy()

    # Upscale
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Blur
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Otsu threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Light erosion
    # kernel = np.ones((2,2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)

    return img


# -------------------------------------------------------
# OCR PIPELINE
# -------------------------------------------------------

def run_ocr(mrz_img):

    # Accept either path or image
    if isinstance(mrz_img, str):
        img = cv2.imread(mrz_img, cv2.IMREAD_GRAYSCALE)
    else:
        img = mrz_img

    if img is None:
        raise RuntimeError("Cannot load MRZ image")

    h = img.shape[0]

    line1 = img[:h//2,:]
    line2 = img[h//2:,:]

    text1 = ocr_line(line1)
    text2 = ocr_line(line2)

    text1 = text1[:44].ljust(44,"<")
    text2 = text2[:44].ljust(44,"<")

    text1,text2 = validate_and_correct_mrz(text1,text2)

    print("\nFinal MRZ")
    print("--------------------------------------------")
    print(text1)
    print(text2)
    print("--------------------------------------------")

    return text1,text2

def save(img, filename):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(path, img)

    print(f"[Save]  {path}")
# -------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------

if __name__ == "__main__":

    mrz_image = os.path.join(OUTPUT_DIR, "mrz_region.png")

    run_ocr(mrz_image)