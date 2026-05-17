import os
from typing import List
from google import genai
from google.genai import types
from enterprice_rag.core.interfaces import EmbeddingProvider
from enterprice_rag.config.settings import GEMINI_API_KEY, EMBED_DIM

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = "gemini-embedding-001"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.client = genai.Client()
        self.model = model

    def embed_text(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        return self.embed_batch([text], task_type=task_type)[0]

    def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        config_task_type = {
            "retrieval_document": "RETRIEVAL_DOCUMENT",
            "retrieval_query": "RETRIEVAL_QUERY",
            "semantic_similarity": "SEMANTIC_SIMILARITY"
        }.get(task_type, "RETRIEVAL_DOCUMENT")

        embeddings = [[0.0] * EMBED_DIM] * len(texts)
        valid_indices = []
        texts_to_send = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                texts_to_send.append(text)

        if not texts_to_send:
            return embeddings

        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts_to_send,
                config=types.EmbedContentConfig(
                    task_type=config_task_type, output_dimensionality=EMBED_DIM
                ),
            ).embeddings

            for i, emb in enumerate(result):
                embeddings[valid_indices[i]] = emb.values
        except Exception as e:
            print(f"Error embedding texts with Gemini: {e}")
        
        return embeddings
