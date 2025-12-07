import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Any
import operator

from processing.retriever import search
from processing.llm_infer import query_ollama, query_ollama_stream
from storage.postgres_client import SessionLocal
from config.settings import DEFAULT_TOP_K
from utils.excel_writer import save_query_to_csv

# ============================================================================
# OPTIMIZED PROMPTS - SHORT & GENERAL
# ============================================================================

QUERY_REWRITE_PROMPT = """Rewrite this search query to be more specific and searchable:

Original: {query}

Rewritten query:"""

QUERY_REFINE_PROMPT = """The previous search didn't work well. Create a better search query.

Previous query: {previous_query}
What was found: {context_summary}

Better query:"""

REFLECTION_PROMPT = """Can this context answer the question? Reply only 'yes' or 'no'.

Question: {query}
Context: {context_preview}

Answer:"""

GENERATION_PROMPT = """Answer the following question based *only* on the provided context.

Question: {query}

Context:
{context}

Answer:"""

# ============================================================================
# STATE
# ============================================================================


class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    context: Annotated[list, operator.add]
    answer: Any
    reflection: str
    iterations: int


# ============================================================================
# NODES
# ============================================================================


def query_analyzer(state):
    """Rewrites query for better search."""
    print("---ANALYZING QUERY---")
    original_query = state["original_query"]
    iterations = state.get("iterations", 0)

    if iterations > 0:
        # Refine based on previous attempt
        previous_query = state["rewritten_query"]
        context = state["context"]  # Only last 3 chunks

        # Create short summary for refinement
        context_summary = " | ".join([c["content"][:100] for c in context])

        prompt = QUERY_REFINE_PROMPT.format(
            previous_query=previous_query, context_summary=context_summary
        )
        rewritten_query = query_ollama(prompt, task="query_rewrite").strip()
    else:
        # First attempt
        prompt = QUERY_REWRITE_PROMPT.format(query=original_query)
        rewritten_query = query_ollama(prompt, task="query_rewrite").strip()

    print(f"Query: {rewritten_query}")
    return {"rewritten_query": rewritten_query, "iterations": iterations + 1}


def retriever(state):
    """Retrieves relevant documents."""
    print("---RETRIEVING---")
    rewritten_query = state["rewritten_query"]

    session = SessionLocal()
    try:
        context = search(session, rewritten_query, top_k=DEFAULT_TOP_K)
        print(f"Found {len(context)} chunks")
    finally:
        session.close()

    return {"context": context}


def reflection(state):
    """Checks if context is good enough."""
    print("---REFLECTING---")
    original_query = state["original_query"]
    context = state["context"]  # Only check recent context

    if not context:
        print("Decision: no (no context found)")
        return {"reflection": "no"}

    # Create short preview (first 200 chars of each chunk)
    context_preview = "\n".join([c["content"] + "..." for c in context])

    prompt = REFLECTION_PROMPT.format(
        query=original_query, context_preview=context_preview
    )

    result = query_ollama(prompt, task="reflection").strip().lower()

    # Robust parsing
    if "yes" in result.split()[0]:
        decision = "yes"
    elif "no" in result.split()[0]:
        decision = "no"
    else:
        # Default to yes after first iteration to avoid loops
        decision = "yes" if state["iterations"] > 0 else "no"

    print(f"Decision: {decision}")
    return {"reflection": decision}


def generator(state):
    """Generates final answer and streams it."""
    print("---GENERATING---")
    original_query = state["original_query"]
    context = state["context"]

    context_str = "\n---\n".join([c["content"] for c in context])
    prompt = GENERATION_PROMPT.format(query=original_query, context=context_str)

    answer_stream = query_ollama_stream(prompt, task="generation")

    def stream_and_save():
        full_answer = ""
        for chunk in answer_stream:
            full_answer += chunk
            yield chunk

        if context:
            source_document = list(set([c["doc_id"] for c in context]))
            save_query_to_csv(original_query, full_answer, ", ".join(source_document))

    return {"answer": stream_and_save()}


# ============================================================================
# ROUTING
# ============================================================================


def should_continue(state):
    """Decides next step based on reflection."""
    if state["reflection"] == "yes":
        return "generate"
    elif state["iterations"] >= 2:  # Max 2 iterations
        return "generate"  # Generate anyway with what we have
    else:
        return "query_analyzer"


# ============================================================================
# GRAPH
# ============================================================================

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

# ============================================================================
# RUNNER
# ============================================================================


def run_agent(query):
    """Run the RAG agent and stream the answer."""
    inputs = {"original_query": query, "iterations": 0}
    for event in app.stream(inputs, {"recursion_limit": 5}):
        if "generator" in event:
            answer_generator = event["generator"]["answer"]
            for chunk in answer_generator:
                yield chunk


if __name__ == "__main__":
    import json

    # The run_agent function is now a generator, so we need to iterate over it
    full_response = ""
    for chunk in run_agent("What are the instructions to tenderers?"):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n" + "=" * 50)
    print("FINAL ANSWER (from runner):")
    print("=" * 50)
    print(full_response)

    # Save output
    output_path = "/home/barry/enterprice_rag/output_files/output_meta.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "query": "What are the instructions to tenderers?",
                "answer": full_response,
            },
            f,
            indent=2,
        )
