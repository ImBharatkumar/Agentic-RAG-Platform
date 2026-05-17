from enterprice_rag.core.factory import get_llm, get_embedding, get_vector_store
from enterprice_rag.config.settings import DEFAULT_TOP_K
from enterprice_rag.utils.excel_writer import save_query_to_csv

# PROMPTS (Preserved from original)
QUERY_REWRITE_PROMPT = """You are a search query optimizer. 
Rewrite the following search query to be highly effective for vector search (semantic retrieval) and keyword search.
Output ONLY the rewritten query and nothing else.

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

def query_analyzer(state):
    print("---ANALYZING QUERY---")
    llm = get_llm()
    original_query = state["original_query"]
    iterations = state.get("iterations", 0)

    if iterations > 0:
        previous_query = state["rewritten_query"]
        context = state["context"]
        context_summary = " | ".join([c["content"][:100] for c in context])
        prompt = QUERY_REFINE_PROMPT.format(previous_query=previous_query, context_summary=context_summary)
        rewritten_query = llm.generate(prompt, task="query_rewrite").strip()
    else:
        prompt = QUERY_REWRITE_PROMPT.format(query=original_query)
        rewritten_query = llm.generate(prompt, task="query_rewrite").strip()

    if not rewritten_query:
        rewritten_query = original_query

    print(f"Query: '{rewritten_query}'")
    return {"rewritten_query": rewritten_query, "iterations": iterations + 1}

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
    context = state["context"]

    context_str = "\n---\n".join([c["content"] for c in context])
    prompt = GENERATION_PROMPT.format(query=original_query, context=context_str)

    answer_stream = llm.stream(prompt, task="generation")

    def stream_and_save():
        full_answer = ""
        for chunk in answer_stream:
            full_answer += chunk
            yield chunk

        if context:
            source_document = list(set([c["doc_id"] for c in context]))
            save_query_to_csv(original_query, full_answer, ", ".join(source_document))

    return {"answer": stream_and_save()}
