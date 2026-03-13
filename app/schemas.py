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
    doc_id: str
    stored_filename: str
    dense_indexing: str


class AskData(BaseModel):
    answer: str
    critique: str | None = None
    intent: str | None = None
