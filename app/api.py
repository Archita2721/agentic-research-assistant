import re
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from constants import DOCUMENTS_DIR
from app.graph import build_graph
from app.schemas import Query
from tools.document_loader import load_document
from tools.text_splitter import split_documents
from vectorstore.faiss_store import add_documents_dense, add_documents_sparse

router = APIRouter()
graph = build_graph()

_uploads_dir = Path(DOCUMENTS_DIR)
_uploads_dir.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return filename or "upload"

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    total_start = time.perf_counter()
    doc_id = uuid4().hex
    original_filename = file.filename or "upload"
    safe_name = _sanitize_filename(original_filename)
    stored_filename = f"{doc_id}_{safe_name}"
    file_path = _uploads_dir / stored_filename

    save_start = time.perf_counter()
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[timing] upload.save_file {(time.perf_counter() - save_start):.3f}s stored={stored_filename}", flush=True)

    load_split_start = time.perf_counter()
    docs = load_document(
        str(file_path),
        doc_id=doc_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=file.content_type,
    )
    chunks = split_documents(docs)
    print(
        f"[timing] upload.load_and_split {(time.perf_counter() - load_split_start):.3f}s docs={len(docs)} chunks={len(chunks)}",
        flush=True,
    )

    sparse_start = time.perf_counter()
    add_documents_sparse(chunks)
    print(f"[timing] upload.index_sparse {(time.perf_counter() - sparse_start):.3f}s", flush=True)
    background_tasks.add_task(add_documents_dense, chunks)

    print(f"[timing] upload.total {(time.perf_counter() - total_start):.3f}s", flush=True)
    return {
        "message": "Document uploaded and indexed successfully",
        "doc_id": doc_id,
        "stored_filename": stored_filename,
        "dense_indexing": "queued",
    }


@router.post("/ask")
def ask_question(query: Query):
    start = time.perf_counter()
    result = graph.invoke({"question": query.question})
    print(f"[timing] ask.total {(time.perf_counter() - start):.3f}s", flush=True)
    return {"answer": result["final_answer"]}
