import os

from env_utils import load_env_file


load_env_file()


class APISettings:
    def __init__(self) -> None:
        self.api_host = os.getenv("API_HOST", "127.0.0.1")
        self.api_port = int(os.getenv("API_PORT", "3000"))
        self.storage_root = os.path.abspath(os.getenv("API_STORAGE_DIR", "storage"))
        self.db_path = os.path.abspath(os.getenv("API_DB_PATH", os.path.join(self.storage_root, "mrz.sqlite3")))
        self.cors_allowed_origins = self._parse_csv_env(
            "CORS_ALLOWED_ORIGINS",
            ["*"],
        )
        self.cors_allowed_methods = self._parse_csv_env(
            "CORS_ALLOWED_METHODS",
            ["*"],
        )
        self.cors_allowed_headers = self._parse_csv_env(
            "CORS_ALLOWED_HEADERS",
            ["*"],
        )
        self.cors_allow_credentials = self._parse_bool_env(
            "CORS_ALLOW_CREDENTIALS",
            False,
        )
        self.llm_api_base_url = os.getenv("LLM_API_BASE_URL", "").strip()
        self.llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        self.llm_model = os.getenv("LLM_MODEL", "").strip()
        self.llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60").strip() or "60")

    @staticmethod
    def _parse_csv_env(name: str, default: list[str]) -> list[str]:
        raw = os.getenv(name, "")
        values = [item.strip() for item in raw.split(",") if item.strip()]
        return values or default[:]

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.storage_root, "uploads")

    @property
    def previews_dir(self) -> str:
        return os.path.join(self.storage_root, "previews")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.storage_root, "reports")

    @property
    def exports_dir(self) -> str:
        return os.path.join(self.storage_root, "exports")


settings = APISettings()
