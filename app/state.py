from typing import TypedDict,List

class AgentState(TypedDict):
    question: str
    search_results: list[str]
    documents: List[str]
    final_answer: str