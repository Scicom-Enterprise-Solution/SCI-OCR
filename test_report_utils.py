import json
import os
import tempfile
import unittest

from report_utils import write_pipeline_report


class TestWritePipelineReport(unittest.TestCase):
    def test_writes_only_prefixed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "sample123")
            report_path = write_pipeline_report(output_dir, {"status": "success"})

            self.assertTrue(report_path.endswith("sample123_report.json"))
            self.assertTrue(os.path.isfile(report_path))
            self.assertFalse(os.path.isfile(os.path.join(output_dir, "report.json")))

            with open(report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["report_file"], "sample123_report.json")


if __name__ == "__main__":
    unittest.main()
