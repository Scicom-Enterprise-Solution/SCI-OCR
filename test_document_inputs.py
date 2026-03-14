import unittest

import cv2
import fitz
import numpy as np

from document_inputs import DocumentInputError, load_document_input


class TestDocumentInputs(unittest.TestCase):
    def test_load_image_bytes(self) -> None:
        image = np.full((8, 10, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)

        self.assertTrue(ok)
        loaded = load_document_input(
            file_bytes=encoded.tobytes(),
            filename="sample.png",
        )

        self.assertEqual(loaded.source_type, "image")
        self.assertEqual(loaded.extension, ".png")
        self.assertEqual(loaded.filename, "sample.png")
        self.assertEqual(loaded.image_bgr.shape[:2], (8, 10))

    def test_load_pdf_bytes(self) -> None:
        doc = fitz.open()
        page = doc.new_page(width=200, height=100)
        page.insert_text((20, 50), "MRZ TEST")
        pdf_bytes = doc.tobytes()
        doc.close()

        loaded = load_document_input(
            file_bytes=pdf_bytes,
            filename="sample.pdf",
            dpi=72,
        )

        self.assertEqual(loaded.source_type, "pdf")
        self.assertEqual(loaded.extension, ".pdf")
        self.assertEqual(loaded.filename, "sample.pdf")
        self.assertGreater(loaded.image_bgr.shape[0], 0)
        self.assertGreater(loaded.image_bgr.shape[1], 0)

    def test_rejects_missing_source(self) -> None:
        with self.assertRaises(DocumentInputError):
            load_document_input(filename="sample.png")

    def test_rejects_ambiguous_source(self) -> None:
        with self.assertRaises(DocumentInputError):
            load_document_input(
                input_path="samples/ww.png",
                file_bytes=b"abc",
                filename="sample.png",
            )


if __name__ == "__main__":
    unittest.main()
