from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.deps import ensure_api_state
from api.routes import extraction, references, uploads
from api.services.document_service import ensure_storage_dirs


ensure_storage_dirs()
ensure_api_state()

app = FastAPI(title="MRZ TD3 API", version="0.1.0")
app.include_router(uploads.router)
app.include_router(extraction.router)
app.include_router(references.router)
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
