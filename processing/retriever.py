import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from processing.embedder import embed_texts
from sqlalchemy import text as sql_text
from storage.postgres_client import SessionLocal
from typing import Any


# ---------- Search helper ----------
def get_embedding_for_query(query: str) -> list[float]:
    return embed_texts([query], task_type="retrieval_query")[0]


def search(session: Session, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    q_emb = get_embedding_for_query(query)
    q_emb_str = str(q_emb)
    # Use cosine distance operator (<=>). We compute 1 - distance to get similarity (0..1)
    sql = sql_text(
        "SELECT id, doc_id, chunk_id, content, metadatas, 1 - (embedding <=> :q) AS score "
        "FROM chunks "
        "ORDER BY embedding <=> :q LIMIT :k"
    )
    # Pass Python list directly and rely on pgvector adapter registered earlier
    rows = session.execute(sql, {"q": q_emb_str, "k": top_k}).fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "doc_id": r[1],
                "chunk_id": r[2],
                "content": r[3],
                "metadatas": r[4],
                "score": float(r[5]),
            }
        )
    return results
