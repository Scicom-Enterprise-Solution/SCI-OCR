import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from fastapi.testclient import TestClient

from api.app import app
from api.services import document_service
from api.services.extraction_service import (
    _relocate_api_report,
    create_extraction,
    fetch_extraction_report_path,
    run_backend_correction,
    should_run_correction,
)


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
            create_extraction(
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
