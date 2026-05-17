from typing import List, Dict, Any
from enterprice_rag.processing.chunker import chunk_text
from enterprice_rag.core.factory import get_embedding, get_vector_store
from enterprice_rag.utils.text_utils import split_into_sentences

def process_document(
    doc_id: str,
    text: str,
    session=None,  # session is now handled inside VectorStoreProvider
    *,
    generate_descriptions: bool = True,
    use_llm_for_description: bool = False,
) -> int:
    """
    Processes a document using sentence-based chunking and stores embeddings in the vector store.
    """
    # 1. Chunking
    chunks_data = chunk_text(text, generate_context_bool=generate_descriptions)
    if not chunks_data:
        print(f"[WARN] No chunks generated for document: {doc_id}")
        return 0

    # 2. Embedding
    embedding_provider = get_embedding()
    contents_to_embed = []
    for chunk in chunks_data:
        situated_content = f"{chunk['context']}\n\n{chunk['content']}" if chunk['context'] else chunk['content']
        contents_to_embed.append(situated_content)
    
    embeddings = embedding_provider.embed_batch(contents_to_embed, task_type="retrieval_document")
    
    for i, emb in enumerate(embeddings):
        chunks_data[i]["embedding"] = emb
        # Add metadata
        chunks_data[i]["metadatas"] = {
            "doc_id": doc_id,
            "chunk_idx": i,
            "sent_count": len(split_into_sentences(chunks_data[i]["content"]))
        }

    # 3. Upserting
    vector_store = get_vector_store()
    inserted = vector_store.upsert(doc_id, chunks_data)

    print(f"[INFO] Inserted {inserted} chunks for document: {doc_id}")
    return inserted
