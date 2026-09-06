from enterprice_rag.providers.llm.ollama import OllamaLLM
from enterprice_rag.providers.embeddings.gemini import GeminiEmbeddingProvider
from enterprice_rag.providers.storage.postgres import PostgresVectorStore

# LLM is cheap to create and task-specific — no singleton needed
def get_llm() -> OllamaLLM:
    return OllamaLLM()

# Embeddings and vector store are expensive — keep singletons
_embedding = None
_vector_store = None

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = GeminiEmbeddingProvider()
    return _embedding

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = PostgresVectorStore()
    return _vector_store
