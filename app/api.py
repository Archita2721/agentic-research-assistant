import shutil

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from app.graph import build_graph
from tools.document_loader import load_document
from tools.text_splitter import split_documents
from vectorstore.faiss_store import create_vectorstore

router = APIRouter()
graph = build_graph()


class Query(BaseModel):
    question: str


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_document(file_path)
    chunks = split_documents(docs)
    create_vectorstore(chunks)

    return {"message": "Document uploaded and indexed successfully"}


@router.post("/ask")
def ask_question(query: Query):
    result = graph.invoke({"question": query.question})
    return {"answer": result["final_answer"]}

