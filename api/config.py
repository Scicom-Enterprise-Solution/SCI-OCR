import os

from env_utils import load_env_file, parse_bool_env, parse_csv_env


class APISettings:
    def __init__(self) -> None:
        object.__setattr__(self, "_overrides", {})

    def _current_values(self) -> dict:
        load_env_file()
        storage_root = os.path.abspath(os.getenv("API_STORAGE_DIR", "storage"))
        return {
            "api_host": os.getenv("API_HOST", "127.0.0.1"),
            "api_port": int(os.getenv("API_PORT", "3000")),
            "storage_root": storage_root,
            "db_path": os.path.abspath(os.getenv("API_DB_PATH", os.path.join(storage_root, "mrz.sqlite3"))),
            "cors_allowed_origins": parse_csv_env("CORS_ALLOWED_ORIGINS", ["*"]),
            "cors_allowed_methods": parse_csv_env("CORS_ALLOWED_METHODS", ["*"]),
            "cors_allowed_headers": parse_csv_env("CORS_ALLOWED_HEADERS", ["*"]),
            "cors_allow_credentials": parse_bool_env("CORS_ALLOW_CREDENTIALS", False),
            "llm_api_base_url": os.getenv("LLM_API_BASE_URL", "").strip(),
            "llm_api_key": os.getenv("LLM_API_KEY", "").strip(),
            "llm_model": os.getenv("LLM_MODEL", "").strip(),
            "llm_timeout_seconds": float(os.getenv("LLM_TIMEOUT_SECONDS", "60").strip() or "60"),
            "uploads_dir": os.path.join(storage_root, "uploads"),
            "previews_dir": os.path.join(storage_root, "previews"),
            "reports_dir": os.path.join(storage_root, "reports"),
            "exports_dir": os.path.join(storage_root, "exports"),
        }

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]

        values = self._current_values()
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._overrides[name] = value

    def __delattr__(self, name: str) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
            return
        raise AttributeError(name)


settings = APISettings()
