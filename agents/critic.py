import time

from llm import chat_llm


def critic_agent(state):
    start = time.perf_counter()

    question = state.get("question", "")
    docs = state.get("documents", [])
    search_results = state.get("search_results", [])
    answer = state.get("final_answer", "")

    context = "\n".join(docs) + "\n" + "\n".join(search_results)

    prompt = f"""
You are a strict self-critic for a research assistant.

Given the question, context, and draft answer:
1) Point out unsupported claims or missing citations from the provided context.
2) Produce a corrected final answer that ONLY uses the provided context.
3) If the context doesn't contain enough info, say: "Not found in the uploaded documents."

Question:
{question}

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

    print(f"[timing] critic.total {(time.perf_counter() - start):.3f}s", flush=True)
    return {"critique": critique, "final_answer": final}

