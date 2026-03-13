import time

from llm import chat_llm


def build_writer_prompt(question: str, docs: list[str], search_results: list[str]) -> str:
    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    return f"""
    Answer the question using the context below. Be concise and direct.

    Question:
    {question}

    Context:
    {context}
    """


def writer_agent(state):
    start = time.perf_counter()
    question = state["question"]
    docs = state.get("documents", [])
    search_results = state.get("search_results", [])
    prompt = build_writer_prompt(question, docs, search_results)
    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    if not context.strip():
        return {"final_answer": "No documents are indexed yet. Upload a document first."}

    print(
        f"[timing] writer.prompt_build {(time.perf_counter() - start):.3f}s question_chars={len(question)} context_chars={len(context)}",
        flush=True,
    )

    llm_start = time.perf_counter()
    response = chat_llm.invoke(prompt)
    print(f"[timing] writer.llm_invoke {(time.perf_counter() - llm_start):.3f}s", flush=True)

    print(f"[timing] writer.total {(time.perf_counter() - start):.3f}s", flush=True)
    return {
        "final_answer": response.content
    }
