from llm import chat_llm

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

    response = chat_llm.invoke(prompt)

    return {
        "final_answer": response.content
    }
