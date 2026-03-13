from typing import Any

from pydantic import BaseModel


class Query(BaseModel):
    question: str


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiResponse(BaseModel):
    ok: bool
    message: str | None = None
    data: Any | None = None
    error: ApiError | None = None
    meta: dict[str, Any] | None = None


class UploadData(BaseModel):
    stored_filename: str
    dense_indexing: str
    job_id: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    extension: str | None = None


class AskData(BaseModel):
    answer: str
    critique: str | None = None
    intent: str | None = None
