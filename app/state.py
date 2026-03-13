from typing import List, NotRequired, TypedDict

class AgentState(TypedDict):
    question: str
    session_id: NotRequired[str]
    search_results: NotRequired[list[str]]
    documents: NotRequired[List[str]]
    final_answer: NotRequired[str]
    route: NotRequired[str]
    critique: NotRequired[str]
    intent: NotRequired[str]
    chat_history: NotRequired[list[dict]]
