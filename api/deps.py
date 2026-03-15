from api.config import settings
from db import init_db


def ensure_api_state() -> None:
    init_db(settings.db_path)
