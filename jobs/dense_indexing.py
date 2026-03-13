from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.documents import Document

from app.mongo import jobs_collection
from vectorstore.faiss_store import add_documents_dense


def run_dense_indexing_job(job_id: str, chunks: list[Document]) -> None:
    jobs = jobs_collection()
    jobs.update_one(
        {"job_id": job_id},
        {"$set": {"status": "running", "started_at": datetime.now(timezone.utc)}},
    )

    try:
        add_documents_dense(chunks)
    except Exception as exc:
        jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc),
                    "error": str(exc),
                }
            },
        )
        raise

    jobs.update_one(
        {"job_id": job_id},
        {"$set": {"status": "done", "finished_at": datetime.now(timezone.utc)}},
    )

