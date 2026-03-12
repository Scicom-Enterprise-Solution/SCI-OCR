import unittest

from run_pipeline import build_stage2_attempts


class TestBuildStage2Attempts(unittest.TestCase):
    def test_pdf_uses_only_aligned_passport(self) -> None:
        attempts = build_stage2_attempts("samples/test.pdf", object(), object())

        self.assertEqual([name for name, _ in attempts], ["aligned_passport"])

    def test_png_uses_original_input_fallback(self) -> None:
        aligned = object()
        original = object()

        attempts = build_stage2_attempts("samples/test.png", aligned, original)

        self.assertEqual(
            [name for name, _ in attempts],
            ["aligned_passport", "original_input"],
        )

    def test_png_skips_duplicate_original_attempt(self) -> None:
        image = object()

        attempts = build_stage2_attempts("samples/test.png", image, image)

        self.assertEqual([name for name, _ in attempts], ["aligned_passport"])


if __name__ == "__main__":
    unittest.main()