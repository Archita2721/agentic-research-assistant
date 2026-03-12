from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from app.constants import OLLAMA_EMBEDDINGS_MODEL

embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)

vectorstore = None


def create_vectorstore(docs):

    global vectorstore

    vectorstore = FAISS.from_documents(
        docs,
        embedding
    )

    return vectorstore


def get_retriever():

    return vectorstore.as_retriever()
