from langgraph.graph import StateGraph, END

from app.state import AgentState

from constants import ENABLE_WEB_SEARCH
from app.enums import Route
from agents.router import router_agent
from agents.smalltalk import smalltalk_agent
from agents.planner import planner_agent
from tools.search_tool import search_agent
from tools.rag_tool import rag_agent
from agents.writer import writer_agent
from agents.critic import critic_agent


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_agent)
    workflow.add_node("smalltalk", smalltalk_agent)
    workflow.add_node("retrieve", rag_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)

    if ENABLE_WEB_SEARCH:
        workflow.add_node("planner", planner_agent)
        workflow.add_node("search", search_agent)
        research_entry = "planner"
        workflow.add_edge("planner", "search")
        workflow.add_edge("search", "retrieve")
    else:
        research_entry = "retrieve"

    workflow.set_entry_point("router")

    def _route(state: AgentState) -> str:
        return state.get("route", Route.RESEARCH.value)

    workflow.add_conditional_edges(
        "router",
        _route,
        {
            Route.SMALLTALK.value: "smalltalk",
            Route.RESEARCH.value: research_entry,
        },
    )

    workflow.add_edge("smalltalk", END)
    workflow.add_edge("retrieve", "writer")
    workflow.add_edge("writer", "critic")

    workflow.add_edge("critic", END)

    return workflow.compile()
