import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from fastapi.testclient import TestClient

from api.app import app
from api.services import document_service
from api.services.extraction_service import _apply_crop, _apply_rotation, _relocate_api_report


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
                "preview_path": "storage/previews/sample.png",
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

            with mock.patch("api.services.extraction_service.settings.reports_dir", reports_dir):
                relocated = _relocate_api_report(source, "doc-1")

            self.assertRegex(
                os.path.basename(relocated),
                r"^doc-1_\d{12}(?:_\d+)?\.json$",
            )
            self.assertTrue(os.path.isfile(relocated))
            self.assertFalse(os.path.exists(source))
