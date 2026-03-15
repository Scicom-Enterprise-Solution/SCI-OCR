import os
import tempfile
import unittest

from db.sqlite import get_document, init_db, insert_document, list_references, upsert_reference


class TestSQLiteDB(unittest.TestCase):
    def test_insert_document_and_reference_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mrz.sqlite3")
            init_db(db_path)

            insert_document(
                db_path,
                {
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
                },
            )

            doc = get_document(db_path, "doc-1")
            self.assertIsNotNone(doc)
            self.assertEqual(doc["filename"], "sample.png")

            saved = upsert_reference(
                db_path,
                {
                    "document_id": "doc-1",
                    "filename": "sample.png",
                    "line1": "P<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                    "line2": "A12345678<UTO8001012M3001012<<<<<<<<<<<<<<06",
                    "notes": "checked",
                    "created_at": "2026-03-15T00:00:00+00:00",
                    "updated_at": "2026-03-15T00:00:00+00:00",
                },
            )
            self.assertEqual(saved["document_id"], "doc-1")
            self.assertEqual(len(list_references(db_path)), 1)
