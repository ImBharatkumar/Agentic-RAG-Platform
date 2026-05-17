import os
from typing import List, Dict
from google import genai
from google.genai import types
from enterprice_rag.config.settings import SENTENCES_PER_CHUNK, GEMINI_API_KEY, EMBED_DIM
from enterprice_rag.utils.text_utils import split_into_sentences
from enterprice_rag.processing.llm_infer import query_ollama

# Configure the generative AI client
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
client = genai.Client()

CONTEXT_PROMPT = """<document>
{doc_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purpose of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def generate_context(doc_text: str, chunk_text: str) -> str:
    """Generates situational context for a chunk using an LLM."""
    prompt = CONTEXT_PROMPT.format(doc_text=doc_text[:10000], chunk_text=chunk_text)  # Limit doc text
    context = query_ollama(prompt, task="context_generation")
    return context


def embed_texts(texts: List[str], task_type="retrieval_document") -> List[List[float]]:
    """Embeds a list of texts using Google's gemini-embedding-001 model."""
    if task_type == "retrieval_document":
        config_task_type = "RETRIEVAL_DOCUMENT"
    elif task_type == "retrieval_query":
        config_task_type = "RETRIEVAL_QUERY"
    else:
        config_task_type = "SEMANTIC_SIMILARITY"

    # Prepare results list
    embeddings = [None] * len(texts)
    valid_indices = []
    texts_to_send = []

    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            texts_to_send.append(text)
        else:
            # Return zero vector for empty/whitespace strings
            embeddings[i] = [0.0] * EMBED_DIM

    if not texts_to_send:
        return embeddings

    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts_to_send,
            config=types.EmbedContentConfig(
                task_type=config_task_type, output_dimensionality=EMBED_DIM
            ),
        ).embeddings

        for i, emb in enumerate(result):
            embeddings[valid_indices[i]] = emb.values

    except Exception as e:
        print(f"DEBUG: Error embedding texts: {e}")
        # Fallback for the whole batch if it fails
        for i in valid_indices:
            embeddings[i] = [0.0] * EMBED_DIM

    return embeddings


def embed_and_chunk_text(
    text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK, generate_context_bool: bool = True
) -> List[Dict]:
    """Splits text into sentences, groups them into chunks, and embeds them."""
    sentences = split_into_sentences(text)

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)

    if not chunks:
        return []

    processed_chunks = []
    texts_to_embed = []
    
    for chunk in chunks:
        context = ""
        if generate_context_bool:
            context = generate_context(text, chunk)
        
        situated_content = f"{context}\n\n{chunk}" if context else chunk
        texts_to_embed.append(situated_content)
        processed_chunks.append({"content": chunk, "context": context})

    chunk_embeddings = embed_texts(texts_to_embed, task_type="retrieval_document")

    for i, embedding in enumerate(chunk_embeddings):
        processed_chunks[i]["embedding"] = embedding

    return processed_chunks
