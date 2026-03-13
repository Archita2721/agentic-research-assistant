from __future__ import annotations

from pathlib import Path
from threading import RLock
import time

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from constants import BOOTSTRAP_DENSE_FROM_DISK, BOOTSTRAP_SPARSE_FROM_MONGO, FAISS_PERSIST_DIR
from llm import embeddings

_lock = RLock()

vectorstore: FAISS | None = None
_all_chunks: list[Document] = []
_bm25: BM25Retriever | None = None
_dense_jobs = 0


def _persist_faiss() -> None:
    if vectorstore is None:
        return
    path = Path(FAISS_PERSIST_DIR)
    path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vectorstore.save_local(str(path))
    print(f"[timing] faiss.save_local {(time.perf_counter() - start):.3f}s path={path}", flush=True)


def add_documents_sparse(docs: list[Document]) -> None:
    global _bm25
    with _lock:
        _all_chunks.extend(docs)
        start = time.perf_counter()
        _bm25 = BM25Retriever.from_documents(_all_chunks)
        print(f"[timing] bm25.build {(time.perf_counter() - start):.3f}s total_chunks={len(_all_chunks)}", flush=True)
        _bm25.k = 6


def add_documents_dense(docs: list[Document]) -> None:
    global _dense_jobs, vectorstore

    with _lock:
        _dense_jobs += 1

    try:
        # Build a separate FAISS shard outside the lock (this is where embeddings happen and can be very slow).
        start = time.perf_counter()
        shard = FAISS.from_documents(docs, embeddings)
        print(f"[timing] faiss.from_documents {(time.perf_counter() - start):.3f}s chunks={len(docs)}", flush=True)

        # Merge quickly under the lock so retrieval isn't blocked for the whole embedding step.
        with _lock:
            if vectorstore is None:
                vectorstore = shard
            else:
                vectorstore.merge_from(shard)

            _persist_faiss()
    finally:
        with _lock:
            _dense_jobs -= 1


def create_vectorstore(docs):
    add_documents_sparse(docs)
    add_documents_dense(docs)
    return vectorstore


def _get_bm25() -> BM25Retriever:
    if _bm25 is None:
        raise RuntimeError("Sparse index is empty. Upload a document first.")
    return _bm25


def get_dense_retriever():
    if vectorstore is None:
        return None
    return vectorstore.as_retriever()


def get_retriever():

    # Backwards-compatible: returns dense retriever if available, else BM25.
    with _lock:
        dense = get_dense_retriever()
        if dense is not None:
            return dense
        return _get_bm25()


def hybrid_retrieve(question: str, *, k_dense: int = 4, k_sparse: int = 6) -> list[Document]:
    results: list[Document] = []

    with _lock:
        # If dense indexing is happening, skip dense retrieval to avoid waiting/contending with merges.
        dense_allowed = _dense_jobs == 0 and vectorstore is not None
        dense = vectorstore.as_retriever() if dense_allowed else None
        sparse = _get_bm25()

    if dense is not None:
        dense.search_kwargs = {"k": k_dense}
        start = time.perf_counter()
        dense_results = dense.invoke(question)
        results.extend(dense_results)
        print(f"[timing] retrieve.dense {(time.perf_counter() - start):.3f}s k={k_dense} got={len(dense_results)}", flush=True)
    else:
        print(f"[timing] retrieve.dense skipped (indexing_in_progress={_dense_jobs > 0})", flush=True)

    sparse.k = k_sparse
    start = time.perf_counter()
    sparse_results = sparse.invoke(question)
    results.extend(sparse_results)
    print(f"[timing] retrieve.sparse {(time.perf_counter() - start):.3f}s k={k_sparse} got={len(sparse_results)}", flush=True)

    # De-duplicate by chunk_id if present, otherwise fallback to text.
    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in results:
        metadata = doc.metadata or {}
        key = str(metadata.get("chunk_id") or "") or doc.page_content
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def bootstrap_indexes() -> None:
    global vectorstore, _bm25, _all_chunks

    if BOOTSTRAP_DENSE_FROM_DISK:
        path = Path(FAISS_PERSIST_DIR)
        if path.exists():
            try:
                start = time.perf_counter()
                vectorstore = FAISS.load_local(
                    str(path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[timing] faiss.load_local {(time.perf_counter() - start):.3f}s path={path}", flush=True)
            except Exception as exc:
                print(f"[warn] faiss.load_local failed: {exc}", flush=True)

    if BOOTSTRAP_SPARSE_FROM_MONGO:
        try:
            from app.mongo import chunks_collection  # local import

            start = time.perf_counter()
            rows = list(chunks_collection().find({}, {"_id": 0, "text": 1, "metadata": 1}))
            docs: list[Document] = []
            for row in rows:
                docs.append(Document(page_content=row.get("text", ""), metadata=row.get("metadata") or {}))

            with _lock:
                _all_chunks = docs
                if _all_chunks:
                    _bm25 = BM25Retriever.from_documents(_all_chunks)
                    _bm25.k = 6

            print(f"[timing] bm25.bootstrap {(time.perf_counter() - start):.3f}s chunks={len(docs)}", flush=True)
        except Exception as exc:
            print(f"[warn] bm25.bootstrap failed: {exc}", flush=True)
