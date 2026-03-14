import os

import fitz
import cv2
import numpy as np
from document_inputs.exceptions import DocumentInputError


def render_pdf_page(pdf_path: str, dpi: int) -> np.ndarray:
    """Render the first page of a PDF to a BGR OpenCV image."""
    if not os.path.isfile(pdf_path):
        raise DocumentInputError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        return _render_first_page(doc, dpi=dpi)
    finally:
        doc.close()


def render_pdf_bytes(pdf_bytes: bytes, dpi: int) -> np.ndarray:
    """Render the first page of uploaded PDF bytes to a BGR OpenCV image."""
    if not pdf_bytes:
        raise DocumentInputError("PDF payload is empty.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return _render_first_page(doc, dpi=dpi)
    finally:
        doc.close()


def _render_first_page(doc: fitz.Document, *, dpi: int) -> np.ndarray:
    if doc.page_count < 1:
        raise DocumentInputError("PDF does not contain any pages.")

    page = doc[0]
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    img_rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
