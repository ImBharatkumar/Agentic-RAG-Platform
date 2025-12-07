from typing import List, Dict
from google import genai
from google.genai import types
import numpy as np
from config.settings import SENTENCES_PER_CHUNK, GEMINI_API_KEY, EMBED_DIM
from utils.text_utils import split_into_sentences
import os

# Configure the generative AI client
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
client = genai.Client()


def embed_texts(texts: List[str], task_type="retrieval_document") -> List[List[float]]:
    """Embeds a list of texts using Google's gemini-embedding-001 model."""
    if not texts:
        return []

    if task_type == "retrieval_document":
        config_task_type = "RETRIEVAL_DOCUMENT"
    elif task_type == "retrieval_query":
        config_task_type = "RETRIEVAL_QUERY"
    else:
        config_task_type = "SEMANTIC_SIMILARITY"

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=config_task_type, output_dimensionality=EMBED_DIM
        ),
    ).embeddings

    return [e.values for e in result]


def embed_and_chunk_text(
    text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK
) -> List[Dict]:
    """Splits text into sentences, groups them into chunks, and embeds them."""
    sentences = split_into_sentences(text)

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)

    if not chunks:
        return []

    chunk_embeddings = embed_texts(chunks, task_type="retrieval_document")

    return [
        {"content": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, chunk_embeddings)
    ]
