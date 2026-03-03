from typing import Any, List, Dict
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from enterprice_rag.processing.embedder import embed_texts


# ---------- Search helpers ----------
def get_embedding_for_query(query: str) -> List[float]:
    return embed_texts([query], task_type="retrieval_query")[0]


def semantic_search(session: Session, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Pure semantic search using pgvector."""
    q_emb = get_embedding_for_query(query)
    q_emb_str = str(q_emb)
    # Use cosine distance operator (<=>). We compute 1 - distance to get similarity (0..1)
    sql = sql_text(
        "SELECT id, doc_id, chunk_id, content, context, metadatas, 1 - (embedding <=> :q) AS score "
        "FROM chunks "
        "ORDER BY embedding <=> :q LIMIT :k"
    )
    rows = session.execute(sql, {"q": q_emb_str, "k": top_k}).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r[0], "doc_id": r[1], "chunk_id": r[2], 
            "content": r[3], "context": r[4], "metadatas": r[5], 
            "score": float(r[6])
        })
    return results


def keyword_search(session: Session, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Keyword search using PostgreSQL full-text search (tsvector)."""
    # Simple plainto_tsquery for the query
    sql = sql_text(
        "SELECT id, doc_id, chunk_id, content, context, metadatas, ts_rank_cd(to_tsvector('english', content || ' ' || COALESCE(context, '')), query) AS score "
        "FROM chunks, plainto_tsquery('english', :q) query "
        "WHERE to_tsvector('english', content || ' ' || COALESCE(context, '')) @@ query "
        "ORDER BY score DESC LIMIT :k"
    )
    rows = session.execute(sql, {"q": query, "k": top_k}).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r[0], "doc_id": r[1], "chunk_id": r[2], 
            "content": r[3], "context": r[4], "metadatas": r[5], 
            "score": float(r[6])
        })
    return results


def reciprocal_rank_fusion(
    search_results_list: List[List[Dict[str, Any]]], k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple search result lists using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1 / (rank + k))
    """
    fused_scores = {}
    docs = {}
    
    for results in search_results_list:
        for rank, res in enumerate(results, start=1):
            doc_id = res["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                docs[doc_id] = res
            fused_scores[doc_id] += 1.0 / (rank + k)
            
    # Sort by fused score
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_id, score in reranked:
        doc = docs[doc_id]
        doc["score"] = score  # RRF score
        final_results.append(doc)
        
    return final_results


def hybrid_search(session: Session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Combines semantic and keyword search result using RRF."""
    # Get 2x top_k from each to allow for overlap and re-ranking
    sem_res = semantic_search(session, query, top_k=top_k * 4)
    key_res = keyword_search(session, query, top_k=top_k * 4)
    
    fused = reciprocal_rank_fusion([sem_res, key_res])
    return fused[:top_k]


# Main entry point (alias for hybrid_search)
def search(session: Session, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return hybrid_search(session, query, top_k=top_k)
