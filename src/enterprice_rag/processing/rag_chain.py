from pathlib import Path
from enterprice_rag.ingestion.pdf_ingest import docling_ocr
from enterprice_rag.storage.vector_store import process_document
from enterprice_rag.processing.retriever import search
from enterprice_rag.storage.postgres_client import SessionLocal
from enterprice_rag.processing.llm_infer import query_ollama_stream
from enterprice_rag.utils.excel_writer import save_query_to_csv

# ============================================================================
# SIMPLE PROMPTS
# ============================================================================

SIMPLE_RAG_PROMPT = """Answer the question using only the provided context. If you can't answer from the context, say "Information not available."

Question: {query}

Context:
{context}

Answer:"""

# ============================================================================
# FUNCTIONS
# ============================================================================


def ingest_and_process(file_paths: list[str]):
    """Ingests and processes a document into vector store."""
    for file_path in file_paths:
        # Ensure file_path is a Path object
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        doc_id = file_path.name
        print(f"Ingesting {doc_id}...")

        content = docling_ocr(file_path)
        if not content:
            print(f"Could not extract text from {file_path}")
            continue

        session = SessionLocal()
        try:
            print(f"Processing and storing {doc_id}...")
            chunks_inserted = process_document(doc_id, content, session)
            print(f"✓ Inserted {chunks_inserted} chunks for {doc_id}")
        finally:
            session.close()


def rag_chain(query: str, top_k: int = 5):
    """Simple RAG: retrieve and generate."""
    session = SessionLocal()
    try:
        print(f"Searching for: '{query}'")
        results = search(session, query, top_k=top_k)

        if not results:
            yield "No relevant documents found."
            return

        print(f"✓ Found {len(results)} relevant chunks")

        # Limit context length
        context = "\n---\n".join([r["content"][:500] for r in results])

        prompt = SIMPLE_RAG_PROMPT.format(query=query, context=context)

        print("Generating answer...")
        answer_stream = query_ollama_stream(prompt, task="generation")

        full_answer = ""
        for chunk in answer_stream:
            full_answer += chunk
            yield full_answer

        # Save the query, answer, and source document
        if results:
            source_document = results[0]["doc_id"]
            save_query_to_csv(query, full_answer, source_document)

    finally:
        session.close()


if __name__ == "__main__":
    # Example 1: Ingest a document
    data_dir = Path("data_files")
    all_files = [file for file in data_dir.iterdir() if file.is_file()]

    # Now call the function and pass all_files as a list of Path objects
    if all_files:
        ingest_and_process(all_files)
    else:
        print(f"No files found in {data_dir} to ingest.")

    # Example 2: Simple RAG query
    # result = rag_chain("What is the environmental policy?")
    # print(result)

    # # Example 3: Use agentic RAG
    from enterprice_rag.agents.langgraph_agent import run_agent

    result = run_agent("Production and productivity in agriculture.")
    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result)
