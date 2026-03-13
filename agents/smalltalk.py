import time

from llm import chat_llm


def build_smalltalk_prompt(user_text: str) -> str:
    return f"""
You are a helpful assistant for a document Q&A app.

The user message is small talk or a greeting. Respond politely in 1-2 short sentences.
Then suggest 2-3 example things they can ask about their uploaded documents.

User message:
{user_text}
"""


def smalltalk_agent(state):
    start = time.perf_counter()
    user_text = (state.get("question") or "").strip()

    prompt = build_smalltalk_prompt(user_text)

    llm_start = time.perf_counter()
    response = chat_llm.invoke(prompt)
    print(f"[timing] smalltalk.llm_invoke {(time.perf_counter() - llm_start):.3f}s", flush=True)
    print(f"[timing] smalltalk.total {(time.perf_counter() - start):.3f}s", flush=True)

    return {"final_answer": response.content}
