from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import parse_node, score_node, planner_node, drafter_node, route_after_planning, human_review_node, finalize_node
from resume import RESUME_TEXT

def score_node_wrapper(state: AgentState) -> AgentState:
    return score_node(state, RESUME_TEXT)

graph = StateGraph(AgentState)
graph.add_node("parse", parse_node)
graph.add_node("score", score_node_wrapper)
graph.add_node("plan", planner_node)
graph.add_node("draft", drafter_node)
graph.add_node("human_review", human_review_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("parse")
graph.add_edge("parse", "score")
graph.add_edge("score", "plan")

graph.add_conditional_edges("plan", route_after_planning, {"draft": "draft", "end": "finalize"})
graph.add_edge("draft", "human_review")
graph.add_edge("human_review", "finalize")
graph.add_edge("finalize", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])