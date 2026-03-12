from llm import chat_llm

def planner_agent(state):
    question = state['question']

    prompt = f"""
    You are a research planner.

    Create a search query to answer this question.

    Question: {question}
    """

    response = chat_llm.invoke(prompt)

    return {
        "search_results": [response.content]
    }
