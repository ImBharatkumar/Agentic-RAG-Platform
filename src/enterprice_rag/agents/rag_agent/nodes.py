from langchain_core.messages import HumanMessage, AIMessage
from enterprice_rag.core.factory import get_llm, get_embedding, get_vector_store
from enterprice_rag.config.settings import DEFAULT_TOP_K
from enterprice_rag.utils.excel_writer import save_query_to_csv

# ── Prompts ────────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """You are a search query optimizer.
Rewrite the following search query to be highly effective for vector search (semantic retrieval) and keyword search.
Output ONLY the rewritten query and nothing else.

Original: {query}

Rewritten query:"""

QUERY_REWRITE_WITH_HISTORY_PROMPT = """You are a search query optimizer with awareness of conversation history.
Use the prior conversation to resolve references like "that", "it", "the same", "section X", etc.
Output ONLY the rewritten, self-contained query — no explanations.

Conversation history (most recent last):
{history}

Current query: {query}

Rewritten query:"""

QUERY_REFINE_PROMPT = """The previous search didn't work well. Create a better search query.

Previous query: {previous_query}
What was found: {context_summary}

Better query:"""

REFLECTION_PROMPT = """Can this context answer the question? Reply only 'yes' or 'no'.

Question: {query}
Context: {context_preview}

Answer:"""

CLASSIFY_PROMPT = """Analyze the conversation history and the user's new query.
Decide if we need to search the database of external documents to answer the query, or if it can be answered using only the conversation history (e.g. conversational responses, follow-up clarification, requests to summarize/explain the previous turn).

Respond with exactly one word:
'retrieve' - if the query requires looking up new information or documents.
'generate' - if the query is conversational (like hello, thanks), a clarification, or asks about the previous answers/context.

Conversation history:
{history}

New Query: {query}

Response:"""

GENERATION_PROMPT = """Answer the following question based *only* on the provided context.

Question: {query}

Context:
{context}

Answer:"""

GENERATION_PROMPT_WITH_HISTORY = """Answer the following question based on the provided context and the prior conversation.
If the context is not sufficient, but the answer is clear from the prior conversation, use the prior conversation to answer.

Conversation history (most recent last):
{history}

Question: {query}

Context:
{context}

Answer:"""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def query_analyzer(state):
    print("---ANALYZING QUERY---")
    llm = get_llm()
    original_query = state["original_query"]
    iterations = state.get("iterations", 0)
    messages = state.get("messages", [])

    if iterations > 0:
        # Iterative refinement: previous retrieval pass didn't satisfy reflection
        previous_query = state["rewritten_query"]
        context = state.get("context", [])
        context_summary = " | ".join([c["content"][:100] for c in context])
        prompt = QUERY_REFINE_PROMPT.format(
            previous_query=previous_query, context_summary=context_summary
        )
        rewritten_query = llm.generate(prompt, task="query_rewrite").strip()
        route = "retrieve"
    else:
        # First pass: use conversation history if available to classify and resolve references
        # Exclude the current HumanMessage (last item) to get prior turns
        history_msgs = messages[:-1][-6:] if len(messages) > 1 else []

        if history_msgs:
            history_str = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:300]}"
                for m in history_msgs
            )
            # Classify whether this query actually needs retrieval or is just conversational / follow-up
            classify_prompt = CLASSIFY_PROMPT.format(history=history_str, query=original_query)
            classification = llm.generate(classify_prompt, task="reflection").strip().lower()
            print(f"Classification result: '{classification}'")

            if "generate" in classification:
                route = "generate"
                rewritten_query = original_query
            else:
                route = "retrieve"
                prompt = QUERY_REWRITE_WITH_HISTORY_PROMPT.format(
                    history=history_str, query=original_query
                )
                rewritten_query = llm.generate(prompt, task="query_rewrite").strip()
        else:
            route = "retrieve"
            prompt = QUERY_REWRITE_PROMPT.format(query=original_query)
            rewritten_query = llm.generate(prompt, task="query_rewrite").strip()

    if not rewritten_query:
        rewritten_query = original_query

    print(f"Query: '{rewritten_query}', Route: '{route}'")
    return {"rewritten_query": rewritten_query, "iterations": iterations + 1, "route": route}


def retriever(state):
    print("---RETRIEVING---")
    rewritten_query = state["rewritten_query"]
    embedding = get_embedding()
    vector_store = get_vector_store()

    query_vector = embedding.embed_text(rewritten_query, task_type="retrieval_query")
    context = vector_store.search_hybrid(rewritten_query, query_vector, top_k=DEFAULT_TOP_K)

    print(f"Found {len(context)} chunks")
    return {"context": context}


def reflection(state):
    print("---REFLECTING---")
    llm = get_llm()
    original_query = state["original_query"]
    context = state["context"]

    if not context:
        return {"reflection": "no"}

    context_preview = "\n".join([c["content"][:200] + "..." for c in context])
    prompt = REFLECTION_PROMPT.format(query=original_query, context_preview=context_preview)
    result = llm.generate(prompt, task="reflection").strip().lower()

    if "yes" in result.split()[0]:
        decision = "yes"
    elif "no" in result.split()[0]:
        decision = "no"
    else:
        decision = "yes" if state["iterations"] > 0 else "no"

    print(f"Decision: {decision}")
    return {"reflection": decision}


def generator(state):
    print("---GENERATING---")
    llm = get_llm()
    original_query = state["original_query"]
    
    # If route is generate, we skip retrieval context to avoid noise
    context = state.get("context", []) if state.get("route") != "generate" else []
    messages = state.get("messages", [])

    history_msgs = messages[:-1][-6:] if len(messages) > 1 else []
    if history_msgs:
        history_str = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history_msgs
        )
        context_str = "\n---\n".join([c["content"] for c in context]) if context else "(No context retrieved)"
        prompt = GENERATION_PROMPT_WITH_HISTORY.format(
            history=history_str, query=original_query, context=context_str
        )
    else:
        context_str = "\n---\n".join([c["content"] for c in context])
        prompt = GENERATION_PROMPT.format(query=original_query, context=context_str)

    # Use blocking generate() so the full answer can be stored in the checkpoint
    full_answer = llm.generate(prompt, task="generation")

    # Persist Q&A to CSV log
    if context:
        source_document = list(set([c["doc_id"] for c in context]))
        save_query_to_csv(original_query, full_answer, ", ".join(source_document))

    return {
        "answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
    }
