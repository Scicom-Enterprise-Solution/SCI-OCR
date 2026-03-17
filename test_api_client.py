import json
import os
import tempfile
import unittest

from scripts.api_client import _write_bytes, _write_json, build_extraction_payload, parse_args


class TestAPIClient(unittest.TestCase):
    def test_build_extraction_payload_omits_optional_correction_when_unspecified(self) -> None:
        payload = build_extraction_payload(
            document_id="doc-1",
            input_mode="raw",
            enable_correction=None,
            use_face_hint=False,
        )
        self.assertEqual(
            payload,
            {
                "document_id": "doc-1",
                "input_mode": "raw",
                "use_face_hint": False,
            },
        )

    def test_build_extraction_payload_keeps_explicit_frontend_controls(self) -> None:
        payload = build_extraction_payload(
            document_id="doc-2",
            input_mode="frontend",
            enable_correction=False,
            use_face_hint=False,
        )
        self.assertEqual(
            payload,
            {
                "document_id": "doc-2",
                "input_mode": "frontend",
                "use_face_hint": False,
                "enable_correction": False,
            },
        )

    def test_parse_args_accepts_manual_control_flags(self) -> None:
        args = parse_args(
            [
                "samples/11.png",
                "--base-url",
                "http://127.0.0.1:3000",
                "--input-mode",
                "frontend",
                "--disable-correction",
            ]
        )
        self.assertEqual(args.file, "samples/11.png")
        self.assertEqual(args.base_url, "http://127.0.0.1:3000")
        self.assertEqual(args.input_mode, "frontend")
        self.assertTrue(args.disable_correction)
        self.assertFalse(args.enable_correction)

    def test_parse_args_accepts_save_path(self) -> None:
        args = parse_args(
            [
                "samples/11.png",
                "--save",
                "tmp/api-run.json",
                "--report-save",
                "tmp/report.json",
            ]
        )
        self.assertEqual(args.save, "tmp/api-run.json")
        self.assertEqual(args.report_save, "tmp/report.json")

    def test_write_json_creates_parent_dirs_and_writes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "api-run.json")
            _write_json(target, {"status": "ok", "value": 3})
            with open(target, "r", encoding="utf-8") as f:
                payload = json.load(f)
        self.assertEqual(payload, {"status": "ok", "value": 3})

    def test_write_bytes_creates_parent_dirs_and_writes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "report.json")
            _write_bytes(target, b"{\"status\":\"ok\"}\n")
            with open(target, "rb") as f:
                payload = f.read()
        self.assertEqual(payload, b"{\"status\":\"ok\"}\n")


if __name__ == "__main__":
    unittest.main()
