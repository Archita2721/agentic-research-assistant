from vectorstore.faiss_store import get_retriever

def rag_agent(state):

    question = state["question"]

    retriever = get_retriever()

    docs = retriever.invoke(question)

    texts = [doc.page_content for doc in docs]

    return {
        "documents": texts
    }