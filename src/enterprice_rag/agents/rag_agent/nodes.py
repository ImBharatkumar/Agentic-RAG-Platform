from enterprice_rag.core.factory import get_llm, get_embedding, get_vector_store
from enterprice_rag.config.settings import DEFAULT_TOP_K
from enterprice_rag.utils.excel_writer import save_query_to_csv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

# ── Prompts ────────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """You are a search query optimizer.
Rewrite the following search query to be highly effective for vector search and keyword search.
Output ONLY the rewritten search keywords/query and nothing else.

Original: {query}

Rewritten query:"""

QUERY_REWRITE_WITH_HISTORY_PROMPT = """You are a search query optimizer aware of conversation history.
Use the prior conversation to resolve ambiguous references (e.g., "it", "that", "the previous section").
Output ONLY the rewritten, self-contained search query.

Conversation history:
{history}

Current query: {query}

Rewritten query:"""

QUERY_REFINE_PROMPT = """The previous search didn't retrieve adequate information. Create an improved search query.
Output ONLY the new search query.

Previous query: {previous_query}
Retrieved summary: {context_summary}

Better query:"""

REFLECTION_PROMPT = """Does the following context contain information relevant to answering the question?
Reply ONLY 'yes' or 'no'.

Question: {query}
Context:
{context_preview}

Answer:"""

CLASSIFY_PROMPT = """Analyze the conversation history and user query.
Determine whether to retrieve external documents or generate a direct conversational response.

Reply ONLY with 'retrieve' or 'generate':
- 'retrieve': requires searching documents or database.
- 'generate': conversational greeting, follow-up clarification, or summary of prior turn.

Conversation history:
{history}

Query: {query}

Response:"""

GENERATION_PROMPT = """Answer the following question based only on the provided context.

Question: {query}

Context:
{context}

Answer:"""

GENERATION_PROMPT_WITH_HISTORY = """Answer the following question based on the provided context and prior conversation.

Conversation history:
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
        previous_query = state.get("rewritten_query", original_query)
        context = state.get("context", [])
        context_summary = " | ".join([c["content"][:80] for c in context[:3]])
        prompt = QUERY_REFINE_PROMPT.format(
            previous_query=previous_query, context_summary=context_summary
        )
        rewritten_query = llm.generate(prompt, task="query_rewrite").strip()
        route = "retrieve"
    else:
        # First pass: use conversation history if available to classify and resolve references
        history_msgs = messages[:-1][-6:] if len(messages) > 1 else []

        if history_msgs:
            history_str = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
                for m in history_msgs
            )
            # Fast classification with 10-token cap
            classify_prompt = CLASSIFY_PROMPT.format(history=history_str, query=original_query)
            classification = llm.generate(classify_prompt, task="classification").strip().lower()
            print(f"Classification result: '{classification}'")

            if "generate" in classification and "retrieve" not in classification:
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
    context = state.get("context", [])

    if not context:
        return {"reflection": "no"}

    # Use top 3 snippets preview with 150 chars each to keep the prompt small & blazing fast
    context_preview = "\n".join([f"- {c['content'][:150]}..." for c in context[:3]])
    prompt = REFLECTION_PROMPT.format(query=original_query, context_preview=context_preview)
    result = llm.generate(prompt, task="reflection").strip().lower()

    if "yes" in result:
        decision = "yes"
    elif "no" in result:
        decision = "no"
    else:
        decision = "yes" if state.get("iterations", 0) > 0 else "no"

    print(f"Decision: {decision}")
    return {"reflection": decision}


def generator(state, config: RunnableConfig):
    print("---GENERATING---")

    llm = get_llm()
    original_query = state["original_query"]
    token_queue = config["configurable"]["token_queue"]

    # If route is generate, we skip retrieval context to avoid noise
    context = (
        state.get("context", [])
        if state.get("route") != "generate"
        else []
    )
    messages = state.get("messages", [])

    history_msgs = messages[:-1][-6:] if len(messages) > 1 else []

    if history_msgs:
        history_str = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history_msgs
        )
        context_str = (
            "\n---\n".join(c["content"] for c in context)
            if context
            else "(No context retrieved)"
        )
        prompt = GENERATION_PROMPT_WITH_HISTORY.format(
            history=history_str,
            query=original_query,
            context=context_str,
        )
    else:
        context_str = "\n---\n".join(c["content"] for c in context)
        prompt = GENERATION_PROMPT.format(
            query=original_query,
            context=context_str,
        )

    chat_model = llm.get_chat_model(task="generation")

    # Stream the final response token-by-token
    full_answer = ""
    for chunk in chat_model.stream([HumanMessage(content=prompt)]):
        if not chunk.content:
            continue

        token = chunk.content
        full_answer += token
        token_queue.put(token)

    full_answer = llm._clean(full_answer)

    # Persist Q&A to CSV log
    if context:
        source_document = list({c["doc_id"] for c in context})
        save_query_to_csv(
            original_query,
            full_answer,
            ", ".join(source_document),
        )

    return {
        "answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
    }
