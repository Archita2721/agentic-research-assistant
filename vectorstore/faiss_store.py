from langchain_community.vectorstores import FAISS

from llm import embeddings

vectorstore = None


def create_vectorstore(docs):

    global vectorstore

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


def get_retriever():

    return vectorstore.as_retriever()
