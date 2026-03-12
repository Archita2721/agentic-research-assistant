from langchain_ollama import ChatOllama

from app.constants import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL)

def writer_agent(state):

    question = state["question"]
    docs = state["documents"]
    search_results = state["search_results"]

    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    prompt = f"""
    Answer the question using the context below.

    Question:
    {question}

    Context:
    {context}
    """

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }
