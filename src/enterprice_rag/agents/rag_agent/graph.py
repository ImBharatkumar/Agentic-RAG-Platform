import queue
import threading
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.callbacks import BaseCallbackHandler
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from enterprice_rag.agents.rag_agent.state import AgentState
from enterprice_rag.agents.rag_agent.nodes import query_analyzer, retriever, reflection, generator
from enterprice_rag.config.settings import DATABASE_URL, MAX_ITERATIONS
from langchain_core.messages import HumanMessage


class StreamingTokenHandler(BaseCallbackHandler):
    """Pushes each LLM token into a thread-safe queue as it is generated."""
    def __init__(self, token_queue: queue.Queue):
        self.token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.token_queue.put(token)


def should_continue(state):
    if state.get("reflection") == "yes":
        return "generate"
    elif state.get("iterations", 0) >= MAX_ITERATIONS:
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

# ── Episodic Memory: PostgresSaver checkpointer ──────────────────────────────
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
    """Stream the RAG agent answer token-by-token using LangChain callbacks."""
    q = queue.Queue()
    handler = StreamingTokenHandler(q)

    config = {
        "configurable": {"thread_id": thread_id, "token_queue": q},
        "recursion_limit": 25,
    }
    inputs = {
        "original_query": query,
        "iterations": 0,
        "reflection": "",
        "context": [],
        "messages": [HumanMessage(content=query)],
    }

    def run_graph():
        try:
            for _ in app.stream(inputs, config):
                pass
        except Exception as e:
            q.put(f"Error: {e}")
        finally:
            q.put(None)  # sentinel — signals end of stream

    threading.Thread(target=run_graph, daemon=True).start()

    while True:
        token = q.get()
        if token is None:
            break
        yield token
