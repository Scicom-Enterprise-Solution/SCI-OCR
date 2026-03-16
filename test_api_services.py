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
    _apply_crop,
    _apply_rotation,
    _relocate_api_report,
    create_extraction,
    fetch_extraction_report_path,
)


class TestAPIServiceHelpers(unittest.TestCase):
    def test_apply_crop_uses_normalized_coordinates(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.3}

        cropped = _apply_crop(image, crop)

        self.assertEqual(cropped.shape[:2], (30, 100))

    def test_apply_rotation_90_changes_axes(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        rotated = _apply_rotation(image, 90)

        self.assertEqual(rotated.shape[:2], (200, 100))


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

    def test_document_preview_route_serves_preview_file(self) -> None:
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            preview_path = os.path.join(tmp, "preview.png")
            image = np.zeros((40, 80, 3), dtype=np.uint8)
            cv2.imwrite(preview_path, image)

            with mock.patch("api.routes.documents.fetch_document_preview_path", return_value=preview_path):
                response = client.get("/api/documents/doc-123/preview")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

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
            "preview_path": "storage/previews/sample_preview.png",
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
            previews_dir = os.path.join(tmp, "previews")
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(previews_dir, exist_ok=True)

            existing = {
                "id": "doc-1",
                "filename": "sample.png",
                "file_hash": "hash-1",
                "source_type": "image",
                "extension": ".png",
                "stored_path": "storage/uploads/doc-1.png",
                "preview_path": "storage/previews/doc-1.png",
                "preview_width": 800,
                "preview_height": 600,
                "created_at": "2026-03-15T00:00:00+00:00",
            }
            image = np.zeros((40, 80, 3), dtype=np.uint8)

            with mock.patch("api.services.document_service.ensure_storage_dirs"), \
                 mock.patch("api.services.document_service.get_document_by_hash", return_value=existing.copy()), \
                 mock.patch("api.services.document_service.settings.storage_root", tmp), \
                 mock.patch(
                     "api.services.document_service.from_repo_relative",
                     side_effect=lambda p: os.path.join(tmp, p.replace("/", os.sep).removeprefix("storage" + os.sep)),
                 ), \
                 mock.patch("api.services.document_service.load_document_input") as load_input:
                load_input.return_value = mock.Mock(image_bgr=image, filename="sample.png")
                record = document_service.create_document(b"same-bytes", "sample.png")

            self.assertEqual(record["id"], "doc-1")
            self.assertTrue(record["deduplicated"])
            self.assertTrue(os.path.isfile(os.path.join(uploads_dir, "doc-1.png")))
            self.assertTrue(os.path.isfile(os.path.join(previews_dir, "doc-1.png")))
            preview = cv2.imread(os.path.join(previews_dir, "doc-1.png"))
            self.assertIsNotNone(preview)


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


class TestExtractionUsesPreviewImage(unittest.TestCase):
    def test_create_extraction_loads_preview_image_as_transform_basis(self) -> None:
        image = np.zeros((40, 80, 3), dtype=np.uint8)

        with mock.patch(
            "api.services.extraction_service.fetch_document",
            return_value={
                "id": "doc-1",
                "filename": "sample.jpg",
                "preview_path": "storage/previews/doc-1.png",
            },
        ), mock.patch(
            "api.services.extraction_service._read_bytes",
            return_value=b"png-bytes",
        ) as read_bytes, mock.patch(
            "api.services.extraction_service.load_document_input",
            return_value=mock.Mock(image_bgr=image),
        ) as load_document_input, mock.patch(
            "api.services.extraction_service._encode_png",
            return_value=b"encoded",
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
                crop=None,
                rotation=0,
                transform=None,
                use_face_hint=False,
            )

        read_bytes.assert_called_once_with("storage/previews/doc-1.png")
        load_document_input.assert_called_once_with(file_bytes=b"png-bytes", filename="doc-1.png")
        self.assertEqual(record["document_id"], "doc-1")
