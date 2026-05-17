from typing import Protocol, List, Dict, Any, Optional, Generator, Union

class LLMProvider(Protocol):
    def generate(self, prompt: str, task: str = "generation", **kwargs) -> str:
        """Generates a response for a given prompt."""
        ...

    def stream(self, prompt: str, task: str = "generation", **kwargs) -> Generator[str, None, None]:
        """Streams a response for a given prompt."""
        ...

class EmbeddingProvider(Protocol):
    def embed_text(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """Embeds a single string into a vector."""
        ...

    def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Embeds a list of strings into vectors."""
        ...

class VectorStoreProvider(Protocol):
    def search(self, query_vector: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Searches for similar vectors in the store."""
        ...

    def search_hybrid(self, query_text: str, query_vector: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Performs a hybrid search (semantic + keyword)."""
        ...

    def upsert(self, doc_id: str, chunks: List[Dict[str, Any]], **kwargs) -> int:
        """Inserts or updates chunks in the store."""
        ...
