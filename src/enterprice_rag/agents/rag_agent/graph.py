from langgraph.graph import StateGraph, END
from enterprice_rag.agents.rag_agent.state import AgentState
from enterprice_rag.agents.rag_agent.nodes import query_analyzer, retriever, reflection, generator

def should_continue(state):
    if state["reflection"] == "yes":
        return "generate"
    elif state["iterations"] >= 2:
        return "generate"
    else:
        return "query_analyzer"

workflow = StateGraph(AgentState)

workflow.add_node("query_analyzer", query_analyzer)
workflow.add_node("retriever", retriever)
workflow.add_node("reflection", reflection)
workflow.add_node("generator", generator)

workflow.set_entry_point("query_analyzer")
workflow.add_edge("query_analyzer", "retriever")
workflow.add_edge("retriever", "reflection")
workflow.add_conditional_edges(
    "reflection",
    should_continue,
    {
        "generate": "generator",
        "query_analyzer": "query_analyzer",
    },
)
workflow.add_edge("generator", END)

app = workflow.compile()

def run_agent(query: str):
    """Run the RAG agent and stream the answer."""
    inputs = {"original_query": query, "iterations": 0}
    for event in app.stream(inputs, {"recursion_limit": 5}):
        if "generator" in event:
            answer_generator = event["generator"]["answer"]
            for chunk in answer_generator:
                yield chunk
