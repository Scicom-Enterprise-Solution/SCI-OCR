import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from api.config import settings
from db import init_db


def main() -> None:
    init_db(settings.db_path)
    print(f"Initialized database: {settings.db_path}")


if __name__ == "__main__":
    main()
