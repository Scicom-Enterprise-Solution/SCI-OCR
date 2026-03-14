import os

import cv2
import numpy as np
from document_inputs.exceptions import DocumentInputError


def load_image_file(image_path: str) -> np.ndarray:
    """Load an image file into a BGR OpenCV image."""
    if not os.path.isfile(image_path):
        raise DocumentInputError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise DocumentInputError(f"Unable to read image: {image_path}")

    return img_bgr


def decode_image_bytes(image_bytes: bytes, *, filename: str = "upload") -> np.ndarray:
    """Decode uploaded image bytes into a BGR OpenCV image."""
    if not image_bytes:
        raise DocumentInputError(f"Image payload is empty: {filename}")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise DocumentInputError(f"Unable to decode image payload: {filename}")

    return img_bgr
