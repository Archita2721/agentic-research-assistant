import time

from constants import GENERIC_DOC_QUESTION_PHRASES, MAX_CHUNK_CHARS, MAX_TOTAL_CONTEXT_CHARS, RETRIEVER_K_SPARSE
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.utils import build_previews


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _is_generic_doc_question(question: str) -> bool:
    q = " ".join((question or "").lower().split())
    if not q:
        return True

    if q in GENERIC_DOC_QUESTION_PHRASES:
        return True

    if len(q) <= 10 and "document" in q:
        return True

    return False


def _resolve_latest_doc_id_for_session(session_id: str) -> str | None:
    from app.mongo import uploads_collection  # local import

    latest = uploads_collection().find_one({"session_id": session_id}, sort=[("created_at", -1)])
    return (latest or {}).get("doc_id")


def _fetch_chunks_from_mongo(*, session_id: str, doc_id: str) -> list[str]:
    from app.mongo import chunks_collection  # local import

    rows = list(
        chunks_collection()
        .find({"session_id": session_id, "doc_id": doc_id}, {"_id": 0, "text": 1, "chunk_index": 1})
        .sort("chunk_index", 1)
    )
    return [row.get("text", "") for row in rows if row.get("text")]


def _retrieve_from_mongo_doc(question: str, *, session_id: str, doc_id: str, k: int) -> list[str]:
    from app.mongo import chunks_collection  # local import

    rows = list(
        chunks_collection()
        .find({"session_id": session_id, "doc_id": doc_id}, {"_id": 0, "text": 1, "metadata": 1, "chunk_index": 1})
        .sort("chunk_index", 1)
    )
    docs = [
        Document(page_content=row.get("text", ""), metadata=row.get("metadata") or {})
        for row in rows
        if row.get("text")
    ]
    if not docs:
        return []

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return [d.page_content for d in retriever.invoke(question)]


def rag_agent(state):
    question = state["question"]
    session_id = state.get("session_id")
    doc_id = _resolve_latest_doc_id_for_session(session_id) if session_id else None

    try:
        if session_id and doc_id and _is_generic_doc_question(question):
            start = time.perf_counter()
            texts = _fetch_chunks_from_mongo(session_id=session_id, doc_id=doc_id)
            print(
                f"[timing] rag.mongo_fetch {(time.perf_counter() - start):.3f}s chunks={len(texts)} doc_id={doc_id}",
                flush=True,
            )

            clipped_texts: list[str] = []
            total = 0
            for text in texts:
                clipped = _clip(text, MAX_CHUNK_CHARS)
                if total + len(clipped) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                clipped_texts.append(clipped)
                total += len(clipped)

            print(f"[timing] rag.context chunks_used={len(clipped_texts)} chars={total}", flush=True)
            previews = build_previews(clipped_texts)
            print(f"[debug] rag.context_preview chunks={len(clipped_texts)} preview={previews}", flush=True)
            for i, text in enumerate(clipped_texts):
                print(f"[debug] rag.context_chunk i={i} chars={len(text or '')} text={text}", flush=True)
            return {"documents": clipped_texts}

        if session_id and doc_id:
            start = time.perf_counter()
            texts = _retrieve_from_mongo_doc(question, session_id=session_id, doc_id=doc_id, k=RETRIEVER_K_SPARSE)
            print(
                f"[timing] rag.mongo_bm25 {(time.perf_counter() - start):.3f}s chunks={len(texts)} doc_id={doc_id}",
                flush=True,
            )

            clipped_texts: list[str] = []
            total = 0
            for text in texts:
                clipped = _clip(text, MAX_CHUNK_CHARS)
                if total + len(clipped) > MAX_TOTAL_CONTEXT_CHARS:
                    break
                clipped_texts.append(clipped)
                total += len(clipped)

            print(f"[timing] rag.context chunks_used={len(clipped_texts)} chars={total}", flush=True)
            previews = build_previews(clipped_texts)
            print(f"[debug] rag.context_preview chunks={len(clipped_texts)} preview={previews}", flush=True)
            for i, text in enumerate(clipped_texts):
                print(f"[debug] rag.context_chunk i={i} chars={len(text or '')} text={text}", flush=True)
            return {"documents": clipped_texts}

        # No session/doc => no safe retrieval target
        return {"documents": []}
    except RuntimeError:
        return {"documents": []}
    except Exception as exc:
        print(f"[warn] rag_agent mongo fetch failed: {exc}", flush=True)
        return {"documents": []}
