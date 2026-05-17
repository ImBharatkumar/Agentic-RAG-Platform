from enterprice_rag.core.interfaces import LLMProvider, EmbeddingProvider, VectorStoreProvider
from enterprice_rag.providers.llm.ollama import OllamaLLM
from enterprice_rag.providers.embeddings.gemini import GeminiEmbeddingProvider
from enterprice_rag.providers.storage.postgres import PostgresVectorStore

class ProviderFactory:
    @staticmethod
    def get_llm_provider(provider_type: str = "ollama") -> LLMProvider:
        if provider_type == "ollama":
            return OllamaLLM()
        # Add other providers here (e.g., gemini, openai)
        raise ValueError(f"Unknown LLM provider type: {provider_type}")

    @staticmethod
    def get_embedding_provider(provider_type: str = "gemini") -> EmbeddingProvider:
        if provider_type == "gemini":
            return GeminiEmbeddingProvider()
        raise ValueError(f"Unknown embedding provider type: {provider_type}")

    @staticmethod
    def get_vector_store_provider(provider_type: str = "postgres") -> VectorStoreProvider:
        if provider_type == "postgres":
            return PostgresVectorStore()
        raise ValueError(f"Unknown vector store provider type: {provider_type}")

# Singleton-like access for convenience
_llm = None
_embedding = None
_vector_store = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ProviderFactory.get_llm_provider()
    return _llm

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = ProviderFactory.get_embedding_provider()
    return _embedding

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = ProviderFactory.get_vector_store_provider()
    return _vector_store
