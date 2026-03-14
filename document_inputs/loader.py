import os

from document_inputs.exceptions import DocumentInputError
from document_inputs.image_input import decode_image_bytes, load_image_file
from document_inputs.models import LoadedDocumentPage
from document_inputs.pdf_input import render_pdf_bytes, render_pdf_page


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def load_document_input(
    *,
    input_path: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    dpi: int = 300,
) -> LoadedDocumentPage:
    """Load a document page from a filesystem path or uploaded file bytes."""
    if bool(input_path) == bool(file_bytes):
        raise DocumentInputError("Provide exactly one of input_path or file_bytes.")

    if input_path:
        filename = filename or os.path.basename(input_path)
        extension = _normalize_extension(filename)
        if extension == ".pdf":
            image_bgr = render_pdf_page(input_path, dpi=dpi)
            return LoadedDocumentPage(image_bgr, "pdf", filename, extension)
        if extension in SUPPORTED_IMAGE_EXTS:
            image_bgr = load_image_file(input_path)
            return LoadedDocumentPage(image_bgr, "image", filename, extension)
        raise DocumentInputError(
            f"Unsupported input type: '{extension}'. Supported: .pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp"
        )

    if not filename:
        raise DocumentInputError("filename is required when loading from file_bytes.")

    extension = _normalize_extension(filename)
    if extension == ".pdf":
        image_bgr = render_pdf_bytes(file_bytes, dpi=dpi)
        return LoadedDocumentPage(image_bgr, "pdf", filename, extension)
    if extension in SUPPORTED_IMAGE_EXTS:
        image_bgr = decode_image_bytes(file_bytes, filename=filename)
        return LoadedDocumentPage(image_bgr, "image", filename, extension)
    raise DocumentInputError(
        f"Unsupported input type: '{extension}'. Supported: .pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp"
    )
