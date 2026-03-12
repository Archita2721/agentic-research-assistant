from langchain_ollama import ChatOllama

from app.constants import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL)

def planner_agent(state):
    question = state['question']

    prompt = f"""
    You are a research planner.

    Create a search query to answer this question.

    Question: {question}
    """

    response = llm.invoke(prompt)

    return {
        "search_results": [response.content]
    }
