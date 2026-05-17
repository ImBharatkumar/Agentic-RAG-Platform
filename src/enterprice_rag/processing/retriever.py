from typing import Any, List, Dict
from enterprice_rag.core.factory import get_embedding, get_vector_store

def search(session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Backward compatible search function."""
    embedding = get_embedding()
    vector_store = get_vector_store()
    
    query_vector = embedding.embed_text(query, task_type="retrieval_query")
    return vector_store.search_hybrid(query, query_vector, top_k=top_k)

def hybrid_search(session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return search(session, query, top_k=top_k)
