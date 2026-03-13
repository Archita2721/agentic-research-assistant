import time

from llm import chat_llm


def _is_generic_doc_question(question: str) -> bool:
    q = " ".join((question or "").lower().split())
    if not q:
        return True

    generic = {
        "summarize",
        "summary",
        "summarize the document",
        "summarise the document",
        "tell me about the document",
        "tell me about document",
        "what is mentioned",
        "what is mentioned in the document",
        "what does the document say",
        "what's in the document",
        "whats in the document",
        "overview",
        "give an overview",
    }
    if q in generic:
        return True

    if len(q) <= 10 and "document" in q:
        return True

    return False


def critic_agent(state):
    start = time.perf_counter()

    question = state.get("question", "")
    docs = state.get("documents", [])
    search_results = state.get("search_results", [])
    chat_history = state.get("chat_history", [])
    answer = state.get("final_answer", "")

    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    # Debug: show what context the critic sees (helps catch "Not found..." mistakes).
    previews = []
    for i, text in enumerate(docs[:6]):
        t = (text or "").replace("\n", " ").strip()
        previews.append(f"{i}: {t[:220]}")
    print(
        f"[debug] critic.context question={question!r} chunks={len(docs)} context_chars={len(context)} preview={previews}",
        flush=True,
    )

    history_text = ""
    if chat_history:
        turns = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                turns.append(f"{role.upper()}: {content}")
        history_text = "\n".join(turns)

    prompt = f"""
You are a strict self-critic for a research assistant.

Given the question, context, and draft answer:
1) Point out unsupported claims or missing citations from the provided context.
2) Produce a corrected final answer that ONLY uses the provided context.
3) If the context is empty, say: "Not found in the uploaded documents."
4) If the user is asking generally what the document contains (summary/overview), and context is non-empty, ALWAYS summarize what is present in the context (even if it is just tables, lists, phone numbers, etc.). Do NOT answer "Not found..." in that case.

Question:
{question}

Conversation so far:
{history_text}

Context:
{context}

Draft Answer:
{answer}

Return format:
CRITIQUE: <short critique>
FINAL: <final answer>
"""

    llm_start = time.perf_counter()
    response = chat_llm.invoke(prompt)
    print(f"[timing] critic.llm_invoke {(time.perf_counter() - llm_start):.3f}s", flush=True)

    text = (response.content or "").strip()
    critique = ""
    final = text

    if "FINAL:" in text:
        parts = text.split("FINAL:", 1)
        left = parts[0].strip()
        final = parts[1].strip()
        if "CRITIQUE:" in left:
            critique = left.split("CRITIQUE:", 1)[1].strip()

    # Guardrail: don't allow "Not found..." for generic document questions when we actually have context.
    if context.strip() and _is_generic_doc_question(question):
        if "not found in the uploaded documents" in (final or "").lower():
            final = answer.strip() or final

    print(f"[timing] critic.total {(time.perf_counter() - start):.3f}s", flush=True)
    return {"critique": critique, "final_answer": final}
