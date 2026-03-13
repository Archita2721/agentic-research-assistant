from __future__ import annotations

from datetime import datetime, timezone

from app.mongo import messages_collection
from constants import MEMORY_MAX_MESSAGES


def fetch_recent_messages(session_id: str, limit: int | None = None) -> list[dict]:
    max_messages = limit or MEMORY_MAX_MESSAGES
    cursor = (
        messages_collection()
        .find({"session_id": session_id}, {"_id": 0})
        .sort("created_at", -1)
        .limit(max_messages)
    )
    # Return oldest -> newest
    return list(reversed(list(cursor)))


def append_message(session_id: str, role: str, content: str, *, intent: str | None = None) -> None:
    messages_collection().insert_one(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "intent": intent,
            "created_at": datetime.now(timezone.utc),
        }
    )


def clear_session(session_id: str) -> int:
    result = messages_collection().delete_many({"session_id": session_id})
    return int(result.deleted_count)

