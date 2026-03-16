from typing import Any

from pydantic import BaseModel, Field


class CropBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class ImageTransform(BaseModel):
    micro_rotation: float = Field(default=0.0, ge=-15.0, le=15.0)
    zoom: float = Field(default=1.0, ge=0.5, le=3.0)
    offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    viewport_width: int | None = Field(default=None, ge=1, le=4000)
    viewport_height: int | None = Field(default=None, ge=1, le=4000)


class ExtractionRequest(BaseModel):
    document_id: str
    crop: CropBox | None = None
    rotation: int = Field(default=0)
    transform: ImageTransform | None = None
    use_face_hint: bool = Field(default=False)


class ReferenceCorrectionRequest(BaseModel):
    document_id: str
    line1: str
    line2: str
    notes: str = ""


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    extension: str
    file_hash: str | None = None
    deduplicated: bool = False
    preview_width: int
    preview_height: int


class ExtractionResponse(BaseModel):
    extraction_id: str
    status: str
    filename: str
    line1: str
    line2: str
    parsed: dict[str, Any]
    duration_ms: float | None = None
    report_path: str | None = None
    document_id: str


class ReferenceCorrectionResponse(BaseModel):
    id: int
    document_id: str
    filename: str
    line1: str
    line2: str
    notes: str
    created_at: str
    updated_at: str
