from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import parse_node, score_node
from resume import RESUME_TEXT

def score_node_wrapper(state: AgentState) -> AgentState:
    return score_node(state, RESUME_TEXT)

graph = StateGraph(AgentState)
graph.add_node("parse", parse_node)
graph.add_node("score", score_node_wrapper)

graph.set_entry_point("parse")
graph.add_edge("parse", "score")
graph.add_edge("score", END)

app = graph.compile()