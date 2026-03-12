import time

from constants import MAX_CHUNK_CHARS, MAX_TOTAL_CONTEXT_CHARS, RETRIEVER_K_DENSE, RETRIEVER_K_SPARSE
from vectorstore.faiss_store import hybrid_retrieve


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"

def rag_agent(state):

    question = state["question"]

    try:
        start = time.perf_counter()
        docs = hybrid_retrieve(question, k_dense=RETRIEVER_K_DENSE, k_sparse=RETRIEVER_K_SPARSE)
        print(f"[timing] rag.retrieve_total {(time.perf_counter() - start):.3f}s docs={len(docs)}", flush=True)
    except RuntimeError:
        return {"documents": []}

    texts: list[str] = []
    total = 0
    for doc in docs:
        clipped = _clip(doc.page_content, MAX_CHUNK_CHARS)
        if total + len(clipped) > MAX_TOTAL_CONTEXT_CHARS:
            break
        texts.append(clipped)
        total += len(clipped)

    print(f"[timing] rag.context chunks_used={len(texts)} chars={total}", flush=True)
    return {
        "documents": texts
    }
