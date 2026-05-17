from typing import List, Dict, Any, Optional
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from enterprice_rag.core.interfaces import VectorStoreProvider
from enterprice_rag.storage.postgres_client import Chunk, SessionLocal

class PostgresVectorStore(VectorStoreProvider):
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    def search(self, query_vector: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        session = self.session_factory()
        try:
            q_emb_str = str(query_vector)
            sql = sql_text(
                "SELECT id, doc_id, chunk_id, content, context, metadatas, 1 - (embedding <=> :q) AS score "
                "FROM chunks "
                "ORDER BY embedding <=> :q LIMIT :k"
            )
            rows = session.execute(sql, {"q": q_emb_str, "k": top_k}).fetchall()
            return [
                {
                    "id": r[0], "doc_id": r[1], "chunk_id": r[2], 
                    "content": r[3], "context": r[4], "metadatas": r[5], 
                    "score": float(r[6])
                } for r in rows
            ]
        finally:
            session.close()

    def search_hybrid(self, query_text: str, query_vector: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        session = self.session_factory()
        try:
            # 1. Semantic search
            sem_res = self.search(query_vector, top_k=top_k * 4)
            
            # 2. Keyword search
            sql = sql_text(
                "SELECT id, doc_id, chunk_id, content, context, metadatas, ts_rank_cd(to_tsvector('english', content || ' ' || COALESCE(context, '')), query) AS score "
                "FROM chunks, plainto_tsquery('english', :q) query "
                "WHERE to_tsvector('english', content || ' ' || COALESCE(context, '')) @@ query "
                "ORDER BY score DESC LIMIT :k"
            )
            rows = session.execute(sql, {"q": query_text, "k": top_k * 4}).fetchall()
            key_res = [
                {
                    "id": r[0], "doc_id": r[1], "chunk_id": r[2], 
                    "content": r[3], "context": r[4], "metadatas": r[5], 
                    "score": float(r[6])
                } for r in rows
            ]

            # 3. Reciprocal Rank Fusion (RRF)
            return self._rrf([sem_res, key_res], k=60)[:top_k]
        finally:
            session.close()

    def upsert(self, doc_id: str, chunks: List[Dict[str, Any]], **kwargs) -> int:
        session = self.session_factory()
        try:
            inserted = 0
            for chunk_idx, chunk_info in enumerate(chunks):
                chunk_content = chunk_info["content"]
                chunk_context = chunk_info.get("context", "")
                chunk_embedding = chunk_info["embedding"]
                
                search_text = f"{chunk_content} {chunk_context}"

                row = Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_idx,
                    content=chunk_content,
                    context=chunk_context,
                    metadatas=chunk_info.get("metadatas", {}),
                    embedding=chunk_embedding,
                    search_vector=search_text
                )
                session.add(row)
                inserted += 1
            session.commit()
            return inserted
        except Exception as e:
            session.rollback()
            print(f"Error upserting chunks: {e}")
            return 0
        finally:
            session.close()

    def _rrf(self, search_results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        fused_scores = {}
        docs = {}
        for results in search_results_list:
            for rank, res in enumerate(results, start=1):
                doc_id = res["id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    docs[doc_id] = res
                fused_scores[doc_id] += 1.0 / (rank + k)
        
        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        for doc_id, score in reranked:
            doc = docs[doc_id]
            doc["score"] = score
            final_results.append(doc)
        return final_results
