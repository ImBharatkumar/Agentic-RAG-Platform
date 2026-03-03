from enterprice_rag.utils.text_utils import split_into_sentences
from enterprice_rag.processing.embedder import embed_and_chunk_text
from enterprice_rag.storage.postgres_client import Chunk


# ---------- Main document processing pipeline ----------
def process_document(
    doc_id: str,
    text: str,
    session,
    *,
    generate_descriptions: bool = True,
    use_llm_for_description: bool = False,
) -> int:
    """
    Processes a document using sentence-based chunking and stores embeddings in Postgres.
    """

    chunks_data = embed_and_chunk_text(text)
    if not chunks_data:
        print(f"[WARN] No chunks generated for document: {doc_id}")
        return 0

    inserted = 0
    for chunk_idx, chunk_info in enumerate(chunks_data):
        chunk_content = chunk_info["content"]
        chunk_context = chunk_info.get("context", "")
        chunk_embedding = chunk_info["embedding"]

        sentence_count = len(split_into_sentences(chunk_content))
        
        # Combined text for keyword search (content + context)
        search_text = f"{chunk_content} {chunk_context}"

        row = Chunk(
            doc_id=doc_id,
            chunk_id=chunk_idx,
            content=chunk_content,
            context=chunk_context,
            metadatas={
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "sent_count": sentence_count,
            },
            embedding=chunk_embedding,
            search_vector=search_text  # Will be converted to tsvector in SQL or via trigger
        )
        session.add(row)
        inserted += 1

    session.commit()
    print(f"[INFO] Inserted {inserted} chunks for document: {doc_id}")
    return inserted
