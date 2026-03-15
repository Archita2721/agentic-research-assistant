import time

from llm import chat_llm
from app.utils import build_previews


def build_writer_prompt(
    question: str,
    docs: list[str],
    search_results: list[str],
    *,
    chat_history: list[dict] | None = None,
) -> str:
    context = "\n".join(docs) + "\n" + "\n".join(search_results)
    history_text = ""
    if chat_history:
        turns = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            turns.append(f"{role.upper()}: {content}")
        if turns:
            history_text = "\n".join(turns)

    return f"""
    Answer the question using the context below. Be concise and direct.

    Conversation so far (may help with references like "it/that"):
    {history_text}

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
    prompt = build_writer_prompt(question, docs, search_results, chat_history=state.get("chat_history"))
    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    if not context.strip():
        return {"final_answer": "No documents are indexed yet. Upload a document first."}

    # Debug: show what context is actually being sent to the LLM.
    previews = build_previews(docs)
    print(
        f"[debug] writer.context question={question!r} chunks={len(docs)} context_chars={len(context)} preview={previews}",
        flush=True,
    )

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
