"""LangGraph graph definition and pipeline state schema."""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph


class PipelineState(TypedDict):
    query: str
    r0: Optional[str]
    r1: Optional[str]
    r2: Optional[dict]
    r2_text: Optional[str]
    error: Optional[str]


def build_graph():
    from agent.formatter import formatter_node
    from agent.orchestrator import orchestrator_node

    graph = StateGraph(PipelineState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("formatter", formatter_node)
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "formatter")
    graph.add_edge("formatter", END)
    return graph.compile()
