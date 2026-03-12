from langgraph.graph import StateGraph, END

from app.state import AgentState

from constants import ENABLE_WEB_SEARCH
from agents.planner import planner_agent
from tools.search_tool import search_agent
from tools.rag_tool import rag_agent
from agents.writer import writer_agent


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", rag_agent)
    workflow.add_node("writer", writer_agent)

    if ENABLE_WEB_SEARCH:
        workflow.add_node("planner", planner_agent)
        workflow.add_node("search", search_agent)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "search")
        workflow.add_edge("search", "retrieve")
    else:
        workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "writer")

    workflow.add_edge("writer", END)

    return workflow.compile()
