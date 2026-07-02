from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from enterprice_rag.agents.rag_agent.state import AgentState
from enterprice_rag.agents.rag_agent.nodes import query_analyzer, retriever, reflection, generator
from enterprice_rag.config.settings import DATABASE_URL
from langchain_core.messages import HumanMessage

def should_continue(state):
    if state["reflection"] == "yes":
        return "generate"
    elif state["iterations"] >= 2:
        return "generate"
    else:
        return "query_analyzer"

def route_query(state):
    return state.get("route", "retrieve")

workflow = StateGraph(AgentState)

workflow.add_node("query_analyzer", query_analyzer)
workflow.add_node("retriever", retriever)
workflow.add_node("reflection", reflection)
workflow.add_node("generator", generator)

workflow.set_entry_point("query_analyzer")
workflow.add_conditional_edges(
    "query_analyzer",
    route_query,
    {
        "retrieve": "retriever",
        "generate": "generator",
    },
)
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

# ── Episodic Memory: PostgresSaver checkpointer ─────────────────────────────
# Strip SQLAlchemy dialect prefix — psycopg needs a plain libpq connection string
_PG_CONN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

# Create a connection pool configured for PostgresSaver checkpointer
pool = ConnectionPool(
    conninfo=_PG_CONN,
    kwargs={"autocommit": True, "row_factory": dict_row},
    max_size=10
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()  # creates langgraph_checkpoint tables once

app = workflow.compile(checkpointer=checkpointer)



def run_agent(query: str, thread_id: str = "default"):
    """Run the RAG agent with episodic memory and stream the answer."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 5}
    inputs = {
        "original_query": query,
        "iterations": 0,
        # Seed messages with the new human turn; checkpointer appends to history
        "messages": [HumanMessage(content=query)],
    }

    for event in app.stream(inputs, config):
        if "generator" in event:
            answer = event["generator"]["answer"]
            if isinstance(answer, str):
                # Yield word-by-word for a streaming feel
                words = answer.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
