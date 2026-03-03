from enterprice_rag.storage.postgres_client import SessionLocal, Chunk
from enterprice_rag.storage.vector_store import process_document
from enterprice_rag.processing.retriever import search
from sqlalchemy import delete

def verify():
    session = SessionLocal()
    
    # 1. Clean up test data
    print("Cleaning up old test data...")
    session.execute(delete(Chunk).where(Chunk.doc_id == "test_verification_doc"))
    session.commit()
    
    # 2. Ingest a sample text
    test_text = """
    The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, 
    the one-mile-wide, three-mile-long strait connecting San Francisco Bay and the Pacific Ocean. 
    It was opened in 1937 and was once the longest and tallest suspension bridge in the world.
    Its signature International Orange color was chosen because it provides high visibility in fog.
    """
    print("Ingesting sample document with Contextual Retrieval...")
    process_document("test_verification_doc", test_text, session)
    
    # 3. Test Keyword Search
    print("\nTesting Keyword Search (visibility fog)...")
    results = search(session, "visibility fog")
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}, Content: {r['content'][:60]}...")
        if r.get('context'):
             print(f"Context: {r['context']}")
    
    # 4. Test Semantic Search
    print("\nTesting Semantic Search (When was the Golden Gate Bridge opened?)...")
    results = search(session, "When was the Golden Gate Bridge opened?")
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}, Content: {r['content'][:60]}...")

    session.close()

if __name__ == "__main__":
    verify()
