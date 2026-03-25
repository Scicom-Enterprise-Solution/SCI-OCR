import os
import tempfile
import unittest

from env_utils import load_env_file
import logger_utils
from api.config import settings


class TestLoadEnvFile(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def _write_temp_env(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".env", text=True)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_ignores_comments_blank_and_invalid_lines(self) -> None:
        path = self._write_temp_env(
            """
# comment

VALID_KEY=ok
INVALID_LINE
ANOTHER = value
            """.strip()
        )
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        load_env_file(path)

        self.assertEqual(os.environ.get("VALID_KEY"), "ok")
        self.assertEqual(os.environ.get("ANOTHER"), "value")
        self.assertNotIn("INVALID_LINE", os.environ)

    def test_strips_wrapping_quotes(self) -> None:
        path = self._write_temp_env(
            """
Q1="double quoted"
Q2='single quoted'
Q3=no_quotes
            """.strip()
        )
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        load_env_file(path)

        self.assertEqual(os.environ.get("Q1"), "double quoted")
        self.assertEqual(os.environ.get("Q2"), "single quoted")
        self.assertEqual(os.environ.get("Q3"), "no_quotes")

    def test_does_not_override_existing_env_var(self) -> None:
        os.environ["EXISTING"] = "original"
        path = self._write_temp_env("EXISTING=from_file")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        load_env_file(path)

        self.assertEqual(os.environ.get("EXISTING"), "original")

    def test_value_can_contain_equals_sign(self) -> None:
        path = self._write_temp_env("TOKEN=a=b=c")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        load_env_file(path)

        self.assertEqual(os.environ.get("TOKEN"), "a=b=c")

    def test_missing_file_is_noop(self) -> None:
        missing_path = os.path.join(tempfile.gettempdir(), "does_not_exist_12345.env")
        if os.path.exists(missing_path):
            os.remove(missing_path)

        load_env_file(missing_path)

        self.assertTrue(True)

    def test_is_debug_enabled_reads_env_dynamically(self) -> None:
        os.environ["DEBUG"] = "true"
        self.assertTrue(logger_utils.is_debug_enabled())

        os.environ["DEBUG"] = "false"
        self.assertFalse(logger_utils.is_debug_enabled())

    def test_api_settings_reads_current_env_dynamically(self) -> None:
        os.environ["API_HOST"] = "127.0.0.1"
        os.environ["API_PORT"] = "3000"
        self.assertEqual(settings.api_host, "127.0.0.1")
        self.assertEqual(settings.api_port, 3000)

        os.environ["API_HOST"] = "0.0.0.0"
        os.environ["API_PORT"] = "9000"
        self.assertEqual(settings.api_host, "0.0.0.0")
        self.assertEqual(settings.api_port, 9000)

    def test_api_settings_override_still_wins_until_removed(self) -> None:
        os.environ["API_PORT"] = "3000"
        settings.api_port = 8123
        self.addCleanup(lambda: hasattr(settings, "_overrides") and settings._overrides.pop("api_port", None))
        self.assertEqual(settings.api_port, 8123)

        del settings.api_port
        self.assertEqual(settings.api_port, 3000)


if __name__ == "__main__":
    unittest.main()
