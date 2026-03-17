import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from api.config import settings


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("uvicorn is required to run the API server.") from exc

    host = os.getenv("API_DEV_HOST", "0.0.0.0").strip() or "0.0.0.0"

    uvicorn.run(
        "api.app:app",
        host=host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
