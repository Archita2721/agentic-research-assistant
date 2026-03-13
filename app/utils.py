import json
import re
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return filename or "upload"


def sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def api_ok(*, data: Any = None, message: str | None = None, meta: dict | None = None) -> dict:
    return {"ok": True, "message": message, "data": data, "error": None, "meta": meta}


def api_error(
    *,
    code: str,
    message: str,
    status_code: int = 400,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "message": None,
            "data": None,
            "error": {"code": code, "message": message, "details": details},
            "meta": None,
        },
    )

