import os
import tempfile
import unittest

from env_utils import load_env_file


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


if __name__ == "__main__":
    unittest.main()
