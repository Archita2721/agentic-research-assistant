import time

from llm import chat_llm

def planner_agent(state):
    start = time.perf_counter()
    question = state['question']

    prompt = f"""
    You are a research planner.

    Create a search query to answer this question.

    Question: {question}
    """

    llm_start = time.perf_counter()
    response = chat_llm.invoke(prompt)
    print(f"[timing] planner.llm_invoke {(time.perf_counter() - llm_start):.3f}s", flush=True)

    print(f"[timing] planner.total {(time.perf_counter() - start):.3f}s", flush=True)
    return {
        "search_results": [response.content]
    }
