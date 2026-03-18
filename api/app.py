from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.deps import ensure_api_state
from api.routes import extraction, llm, references, uploads
from api.services.document_service import ensure_storage_dirs


ensure_storage_dirs()
ensure_api_state()

app = FastAPI(title="MRZ TD3 API", version="0.1.0")

allow_all_origins = "*" in settings.cors_allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else settings.cors_allowed_origins,
    allow_credentials=False if allow_all_origins else settings.cors_allow_credentials,
    allow_methods=settings.cors_allowed_methods,
    allow_headers=settings.cors_allowed_headers,
)

app.include_router(uploads.router)
app.include_router(extraction.router)
app.include_router(llm.router)
app.include_router(references.router)
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
