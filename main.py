from fastapi import FastAPI

from app.api import router
from vectorstore.faiss_store import bootstrap_indexes

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def _bootstrap() -> None:
    bootstrap_indexes()
