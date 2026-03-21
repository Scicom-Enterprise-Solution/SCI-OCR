import os
import tempfile
import unittest
from unittest import mock
from urllib import error

import cv2
import numpy as np

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import app
from api.routes.llm import chat, llm_health
from api.schemas import LLMChatRequest
from api.services import document_service
from api.services.extraction_service import (
    _relocate_api_report,
    create_extraction,
    fetch_extraction_report_path,
    run_backend_correction,
    should_run_correction,
)
from api.services.llm_service import _extract_content, run_chat_completion


class TestAPIRoutes(unittest.TestCase):
    def test_upload_route_maps_document_id_field(self) -> None:
        client = TestClient(app)

        with mock.patch("api.routes.uploads.create_document") as create_document:
            create_document.return_value = {
                "id": "doc-123",
                "filename": "sample.png",
                "file_hash": "abc123",
                "deduplicated": False,
                "source_type": "image",
                "extension": ".png",
                "preview_width": 800,
                "preview_height": 600,
            }

            response = client.post(
                "/api/uploads",
                files={"file": ("sample.png", b"123", "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_id"], "doc-123")
        self.assertEqual(payload["filename"], "sample.png")
        self.assertEqual(payload["file_hash"], "abc123")
        self.assertFalse(payload["deduplicated"])
        self.assertNotIn("preview_path", payload)

    def test_extraction_report_route_serves_report_file(self) -> None:
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("{}")

            with mock.patch("api.routes.extraction.fetch_extraction_report_path", return_value=report_path):
                response = client.get("/api/extractions/ext-123/report")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_llm_health_route_reflects_configuration_state(self) -> None:
        with mock.patch("api.routes.llm.is_llm_enabled", return_value=True), \
             mock.patch("api.routes.llm.settings.llm_api_base_url", "http://127.0.0.1:11434/v1"), \
             mock.patch("api.routes.llm.settings.llm_model", "llama3.1:8b"):
            response = llm_health()

        self.assertEqual(
            response,
            {"enabled": True, "base_url": True, "model": True},
        )

    def test_llm_chat_route_returns_provider_response(self) -> None:
        with mock.patch(
            "api.routes.llm.run_chat_completion",
            return_value={
                "provider": "http://127.0.0.1:11434/v1",
                "model": "llama3.1:8b",
                "content": "Checksum mismatches reduce trust in line 2 parsing.",
                "raw": {"choices": [{"message": {"content": "Checksum mismatches reduce trust in line 2 parsing."}}]},
            },
        ) as run_chat_completion_mock:
            response = chat(
                LLMChatRequest(
                    messages=[
                        {"role": "user", "content": "Explain checksum mismatches."},
                    ]
                )
            )

        self.assertEqual(
            response.content,
            "Checksum mismatches reduce trust in line 2 parsing.",
        )
        _, kwargs = run_chat_completion_mock.call_args
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "Explain checksum mismatches."}],
        )

    def test_llm_chat_route_maps_not_configured_to_503(self) -> None:
        with mock.patch(
            "api.routes.llm.run_chat_completion",
            side_effect=ValueError("LLM integration is not configured"),
        ):
            with self.assertRaises(HTTPException) as exc:
                chat(
                    LLMChatRequest(
                        messages=[
                        {"role": "user", "content": "Hello"},
                        ]
                    )
                )

        self.assertEqual(exc.exception.status_code, 503)
        self.assertEqual(exc.exception.detail, "LLM integration is not configured")


class TestDocumentDeduplication(unittest.TestCase):
    def test_create_document_returns_existing_record_for_same_hash(self) -> None:
        existing = {
            "id": "doc-1",
            "filename": "sample.png",
            "file_hash": "hash-1",
            "source_type": "image",
            "extension": ".png",
            "stored_path": "storage/uploads/sample.png",
            "preview_path": "storage/uploads/sample.png",
            "preview_width": 800,
            "preview_height": 600,
            "created_at": "2026-03-15T00:00:00+00:00",
        }

        with mock.patch("api.services.document_service.ensure_storage_dirs"), \
             mock.patch("api.services.document_service.get_document_by_hash", return_value=existing), \
             mock.patch("api.services.document_service._document_artifacts_exist", return_value=True), \
             mock.patch("api.services.document_service._write_bytes") as write_bytes, \
             mock.patch("api.services.document_service.load_document_input") as load_input:
            record = document_service.create_document(b"same-bytes", "sample.png")

        self.assertEqual(record["id"], "doc-1")
        self.assertTrue(record["deduplicated"])
        write_bytes.assert_not_called()
        load_input.assert_not_called()

    def test_create_document_restores_missing_artifacts_for_existing_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads_dir = os.path.join(tmp, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            existing = {
                "id": "doc-1",
                "filename": "sample.png",
                "file_hash": "hash-1",
                "source_type": "image",
                "extension": ".png",
                "stored_path": "storage/uploads/doc-1.png",
                "preview_path": "storage/uploads/doc-1.png",
                "preview_width": 800,
                "preview_height": 600,
                "created_at": "2026-03-15T00:00:00+00:00",
            }

            with mock.patch("api.services.document_service.ensure_storage_dirs"), \
                 mock.patch("api.services.document_service.get_document_by_hash", return_value=existing.copy()), \
                 mock.patch("api.services.document_service.settings.storage_root", tmp), \
                 mock.patch(
                     "api.services.document_service.from_repo_relative",
                     side_effect=lambda p: os.path.join(tmp, p.replace("/", os.sep).removeprefix("storage" + os.sep)),
                 ):
                record = document_service.create_document(b"same-bytes", "sample.png")

            self.assertEqual(record["id"], "doc-1")
            self.assertTrue(record["deduplicated"])
            self.assertTrue(os.path.isfile(os.path.join(uploads_dir, "doc-1.png")))
            self.assertEqual(record["preview_path"], "storage/uploads/doc-1.png")

    def test_create_document_uses_stored_file_as_preview_basis(self) -> None:
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        with mock.patch("api.services.document_service.ensure_storage_dirs"), \
             mock.patch("api.services.document_service.get_document_by_hash", return_value=None), \
             mock.patch("api.services.document_service._write_bytes"), \
             mock.patch("api.services.document_service.insert_document", side_effect=lambda db_path, record: record), \
             mock.patch("api.services.document_service.load_document_input") as load_input:
            load_input.return_value = mock.Mock(
                image_bgr=image,
                source_type="image",
                extension=".png",
            )
            record = document_service.create_document(b"new-bytes", "sample.png")

        self.assertEqual(record["preview_path"], record["stored_path"])
        self.assertEqual(record["preview_width"], 80)
        self.assertEqual(record["preview_height"], 40)


class TestAPIReportRelocation(unittest.TestCase):
    def test_relocate_api_report_moves_report_to_storage_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "old_report.json")
            reports_dir = os.path.join(tmp, "reports")
            with open(source, "w", encoding="utf-8") as f:
                f.write("{}")

            with mock.patch.object(
                type(_relocate_api_report.__globals__["settings"]),
                "reports_dir",
                new_callable=mock.PropertyMock,
                return_value=reports_dir,
            ):
                relocated = _relocate_api_report(source, "doc-1")

            self.assertRegex(
                os.path.basename(relocated),
                r"^doc-1_\d{12}(?:_\d+)?\.json$",
            )
            self.assertTrue(os.path.isfile(relocated))
            self.assertFalse(os.path.exists(source))

    def test_fetch_extraction_report_path_resolves_repo_relative_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("{}")

            with mock.patch(
                "api.services.extraction_service.fetch_extraction",
                return_value={"id": "ext-1", "report_path": "storage/reports/report.json"},
            ), mock.patch(
                "api.services.extraction_service.from_repo_relative",
                return_value=report_path,
            ):
                resolved = fetch_extraction_report_path("ext-1")

        self.assertEqual(resolved, report_path)


class TestExtractionModes(unittest.TestCase):
    def test_should_run_correction_matches_mode_rules(self) -> None:
        self.assertTrue(should_run_correction("raw", None))
        self.assertTrue(should_run_correction("raw", False))
        self.assertTrue(should_run_correction("frontend", True))
        self.assertFalse(should_run_correction("frontend", None))
        self.assertFalse(should_run_correction("frontend", False))

    def test_create_extraction_frontend_mode_uses_stored_image_without_correction(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.png",
            },
        ), mock.patch(
            "api.services.extraction_service._read_bytes",
            return_value=b"final-bytes",
        ) as read_bytes, mock.patch(
            "api.services.extraction_service._validate_frontend_input",
        ) as validate_frontend_input, mock.patch(
            "api.services.extraction_service.run_backend_correction",
        ) as run_backend_correction, mock.patch(
            "api.services.extraction_service.process_document",
            return_value={
                "status": "success",
                "mrz": {
                    "text": {},
                    "parsed": {},
                    "ocr": {"confidence": {"final_score": 0.91, "warnings": []}},
                },
                "duration_ms": 1.0,
            },
        ) as process_document, mock.patch(
            "api.services.extraction_service.insert_extraction",
            side_effect=lambda db_path, record: record,
        ), mock.patch(
            "api.services.extraction_service._relocate_api_report",
            return_value=None,
        ):
            record = create_extraction(
                document_id="doc-1",
                input_mode="frontend",
                enable_correction=False,
                crop=None,
                rotation=0,
                transform=None,
                use_face_hint=False,
            )

        read_bytes.assert_called_once_with("storage/uploads/doc-1.png")
        validate_frontend_input.assert_called_once_with(b"final-bytes", "sample.jpg", "frontend")
        run_backend_correction.assert_not_called()
        process_document.assert_called_once()
        _, kwargs = process_document.call_args
        self.assertEqual(kwargs["file_bytes"], b"final-bytes")
        self.assertTrue(kwargs["skip_document_alignment"])
        self.assertTrue(kwargs["strict_input_orientation"])
        self.assertTrue(kwargs["prealigned_input"])
        self.assertFalse(kwargs["use_face_hint"])
        self.assertEqual(record["document_id"], "doc-1")
        self.assertFalse(record["enable_correction"])
        self.assertFalse(record["correction_applied"])
        self.assertEqual(record["rotation"], 0)
        self.assertIsNone(record["transform"])
        self.assertIsNone(record["crop"])
        self.assertEqual(record["confidence"], {"final_score": 0.91, "warnings": []})


class TestLLMService(unittest.TestCase):
    def test_extract_content_supports_string_message_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Use line 2 checksums as the stronger signal.",
                    }
                }
            ]
        }

        self.assertEqual(
            _extract_content(payload),
            "Use line 2 checksums as the stronger signal.",
        )

    def test_extract_content_supports_text_parts(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Line 2 is checksum-backed."},
                            {"type": "text", "text": "Treat it as structurally stronger."},
                        ],
                    }
                }
            ]
        }

        self.assertEqual(
            _extract_content(payload),
            "Line 2 is checksum-backed.\nTreat it as structurally stronger.",
        )

    def test_run_chat_completion_sends_openai_compatible_request(self) -> None:
        response_payload = {
            "model": "llama3.1:8b",
            "choices": [
                {
                    "message": {
                        "content": "MRZ line 2 is more trustworthy because checksums constrain it.",
                    }
                }
            ],
        }

        response_mock = mock.MagicMock()
        response_mock.read.return_value = str(response_payload).replace("'", '"').encode("utf-8")
        response_mock.__enter__.return_value = response_mock
        response_mock.__exit__.return_value = False

        with mock.patch("api.services.llm_service.is_llm_enabled", return_value=True), \
             mock.patch("api.services.llm_service.settings.llm_api_base_url", "http://127.0.0.1:11434/v1"), \
             mock.patch("api.services.llm_service.settings.llm_model", "llama3.1:8b"), \
             mock.patch("api.services.llm_service.settings.llm_api_key", ""), \
             mock.patch("api.services.llm_service.settings.llm_timeout_seconds", 12.0), \
             mock.patch("api.services.llm_service.request.urlopen", return_value=response_mock) as urlopen_mock:
            result = run_chat_completion(
                messages=[{"role": "user", "content": "Why is line 2 stronger?"}],
                temperature=0.2,
                max_tokens=120,
            )

        self.assertEqual(result["model"], "llama3.1:8b")
        self.assertIn("checksum", result["content"].lower())
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual(sent_request.full_url, "http://127.0.0.1:11434/v1/chat/completions")

    def test_run_chat_completion_surfaces_http_errors(self) -> None:
        http_error = error.HTTPError(
            url="http://127.0.0.1:11434/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=mock.Mock(read=mock.Mock(return_value=b'{"error":"boom"}')),
        )

        with mock.patch("api.services.llm_service.is_llm_enabled", return_value=True), \
             mock.patch("api.services.llm_service.settings.llm_api_base_url", "http://127.0.0.1:11434/v1"), \
             mock.patch("api.services.llm_service.settings.llm_model", "llama3.1:8b"), \
             mock.patch("api.services.llm_service.settings.llm_api_key", ""), \
             mock.patch("api.services.llm_service.settings.llm_timeout_seconds", 12.0), \
             mock.patch("api.services.llm_service.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(RuntimeError, "LLM request failed"):
                run_chat_completion(messages=[{"role": "user", "content": "Hello"}])

    def test_create_extraction_frontend_mode_rejects_crop(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.png",
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "frontend mode does not accept crop/transform/rotation; send final prepared image instead",
            ):
                create_extraction(
                    document_id="doc-1",
                    input_mode="frontend",
                    enable_correction=False,
                    crop={"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5},
                    rotation=0,
                    transform=None,
                    use_face_hint=False,
                )

    def test_create_extraction_frontend_mode_rejects_transform(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.png",
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "frontend mode does not accept crop/transform/rotation; send final prepared image instead",
            ):
                create_extraction(
                    document_id="doc-1",
                    input_mode="frontend",
                    enable_correction=False,
                    crop=None,
                    rotation=0,
                    transform={"micro_rotation": 1.0},
                    use_face_hint=False,
                )

    def test_create_extraction_frontend_mode_rejects_nonzero_rotation(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.png",
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "frontend mode does not accept crop/transform/rotation; send final prepared image instead",
            ):
                create_extraction(
                    document_id="doc-1",
                    input_mode="frontend",
                    enable_correction=False,
                    crop=None,
                    rotation=90,
                    transform=None,
                    use_face_hint=False,
                )

    def test_create_extraction_raw_mode_runs_correction_by_default(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.jpg",
            },
        ), mock.patch(
            "api.services.extraction_service._read_bytes",
            return_value=b"raw-bytes",
        ), mock.patch(
            "api.services.extraction_service._validate_frontend_input",
        ) as validate_frontend_input, mock.patch(
            "api.services.extraction_service.run_backend_correction",
            return_value=b"corrected-bytes",
        ) as run_backend_correction, mock.patch(
            "api.services.extraction_service.process_document",
            return_value={"status": "success", "mrz": {"text": {}, "parsed": {}}, "duration_ms": 1.0},
        ) as process_document, mock.patch(
            "api.services.extraction_service.insert_extraction",
            side_effect=lambda db_path, record: record,
        ), mock.patch(
            "api.services.extraction_service._relocate_api_report",
            return_value=None,
        ):
            record = create_extraction(
                document_id="doc-1",
                input_mode="raw",
                enable_correction=None,
                crop=None,
                rotation=0,
                transform=None,
                use_face_hint=True,
            )

        validate_frontend_input.assert_called_once_with(b"raw-bytes", "sample.jpg", "raw")
        run_backend_correction.assert_called_once_with(
            b"raw-bytes",
            filename="sample.jpg",
            rotation=0,
            transform=None,
            crop=None,
        )
        _, kwargs = process_document.call_args
        self.assertEqual(kwargs["file_bytes"], b"corrected-bytes")
        self.assertFalse(kwargs["skip_document_alignment"])
        self.assertFalse(kwargs["strict_input_orientation"])
        self.assertFalse(kwargs["prealigned_input"])
        self.assertTrue(kwargs["use_face_hint"])
        self.assertIsNone(record["enable_correction"])
        self.assertTrue(record["correction_applied"])
        self.assertEqual(record["rotation"], 0)
        self.assertIsNone(record["transform"])
        self.assertIsNone(record["crop"])

    def test_create_extraction_raw_mode_accepts_correction_inputs(self) -> None:
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "stored_path": "storage/uploads/doc-1.jpg",
            },
        ), mock.patch(
            "api.services.extraction_service._read_bytes",
            return_value=b"raw-bytes",
        ), mock.patch(
            "api.services.extraction_service._validate_frontend_input",
        ), mock.patch(
            "api.services.extraction_service.run_backend_correction",
            return_value=b"corrected-bytes",
        ), mock.patch(
            "api.services.extraction_service.process_document",
            return_value={"status": "success", "mrz": {"text": {}, "parsed": {}}, "duration_ms": 1.0},
        ), mock.patch(
            "api.services.extraction_service.insert_extraction",
            side_effect=lambda db_path, record: record,
        ), mock.patch(
            "api.services.extraction_service._relocate_api_report",
            return_value=None,
        ):
            record = create_extraction(
                document_id="doc-1",
                input_mode="raw",
                enable_correction=True,
                crop={"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5},
                rotation=90,
                transform={"micro_rotation": 1.0},
                use_face_hint=False,
            )

        self.assertEqual(record["rotation"], 90)
        self.assertEqual(record["transform"], {"micro_rotation": 1.0})
        self.assertEqual(record["crop"], {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5})

    def test_create_extraction_rejects_small_frontend_input(self) -> None:
        image = np.zeros((399, 599, 3), dtype=np.uint8)
        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "small.png",
                "stored_path": "storage/uploads/doc-1.png",
            },
        ), mock.patch(
            "api.services.extraction_service._read_bytes",
            return_value=b"small-bytes",
        ), mock.patch(
            "api.services.extraction_service.load_document_input",
            return_value=mock.Mock(image_bgr=image),
        ):
            with self.assertRaisesRegex(ValueError, "minimum size is 600x400"):
                create_extraction(
                    document_id="doc-1",
                    input_mode="frontend",
                    enable_correction=False,
                    crop=None,
                    rotation=0,
                    transform=None,
                    use_face_hint=False,
                )

    def test_run_backend_correction_is_not_noop_when_rotation_applied(self) -> None:
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[0, 0] = (255, 0, 0)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        corrected = run_backend_correction(
            encoded.tobytes(),
            filename="sample.png",
            rotation=90,
            transform=None,
            crop=None,
        )
        self.assertNotEqual(corrected, encoded.tobytes())

        decoded = cv2.imdecode(np.frombuffer(corrected, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (5, 3))
