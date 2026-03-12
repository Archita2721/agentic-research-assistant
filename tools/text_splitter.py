from langchain_text_splitters import RecursiveCharacterTextSplitter

from constants import TEXT_SPLITTER_CHUNK_OVERLAP, TEXT_SPLITTER_CHUNK_SIZE

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    per_doc_counts: dict[str, int] = {}
    for global_index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        doc_id = str(chunk.metadata.get("doc_id") or "")
        if doc_id:
            per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
            chunk_index = per_doc_counts[doc_id] - 1
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = f"{doc_id}:{chunk_index}"
        else:
            chunk.metadata["chunk_index"] = global_index

    return chunks
