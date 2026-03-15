import os

from env_utils import load_env_file


load_env_file()


class APISettings:
    def __init__(self) -> None:
        self.api_host = os.getenv("API_HOST", "127.0.0.1")
        self.api_port = int(os.getenv("API_PORT", "3000"))
        self.storage_root = os.path.abspath(os.getenv("API_STORAGE_DIR", "storage"))
        self.db_path = os.path.abspath(os.getenv("API_DB_PATH", os.path.join(self.storage_root, "mrz.sqlite3")))

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
