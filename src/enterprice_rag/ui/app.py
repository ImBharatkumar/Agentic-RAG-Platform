import os
import tempfile
from pathlib import Path
import gradio as gr
from enterprice_rag.agents.langgraph_agent import run_agent
from enterprice_rag.ingestion.pdf_ingest import docling_ocr
from enterprice_rag.storage.vector_store import process_document
from enterprice_rag.storage.postgres_client import SessionLocal


def answer_question(query):
    """
    This function takes a user query, passes it to the RAG agent, and streams the response.
    """
    try:
        full_answer = ""
        for chunk in run_agent(query):
            full_answer += chunk
            yield full_answer
    except Exception as e:
        yield f"An error occurred: {e}"


def ingest_files(files: list[str]):
    """
    This function takes a list of files, ingests them into the vector store, and returns a status message.
    """
    if not files:
        return "Please upload one or more files."

    statuses = []

    for file in files:
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix
            ) as tmp:
                with open(file.name, "rb") as f:
                    tmp.write(f.read())

                tmp_path = tmp.name

            markdown_text = docling_ocr(tmp_path)

            if not markdown_text:
                statuses.append(f"Failed to extract text from {file.name}.")

                continue

            session = SessionLocal()

            try:
                doc_id = Path(file.name).stem

                num_chunks = process_document(doc_id, markdown_text, session)

                statuses.append(
                    f"Successfully ingested {num_chunks} chunks from {file.name}."
                )

            finally:
                session.close()

                os.unlink(tmp_path)

        except Exception as e:
            statuses.append(f"An error occurred during ingestion of {file.name}: {e}")

    return "\n".join(statuses)


# Create the Gradio interface


chat_interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(lines=20, label="Answer"),
    title="Enterprise RAG System",
    description="Ask any question and the RAG system will answer it based on the provided documents.",
)


ingestion_interface = gr.Interface(
    fn=ingest_files,
    inputs=gr.File(label="Upload Documents", file_count="multiple"),
    outputs="text",
    title="File Ingestion",
    description="Upload one or more documents to ingest into the RAG system.",
)


iface = gr.TabbedInterface(
    [chat_interface, ingestion_interface], ["Chat", "Ingest File"]
)
# iface = ingestion_interface


if __name__ == "__main__":
    iface.launch()
