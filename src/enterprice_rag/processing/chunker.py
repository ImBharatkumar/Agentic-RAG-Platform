from typing import List, Dict
from enterprice_rag.utils.text_utils import split_into_sentences
from enterprice_rag.core.factory import get_llm
from enterprice_rag.config.settings import SENTENCES_PER_CHUNK

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
    llm = get_llm()
    prompt = CONTEXT_PROMPT.format(doc_text=doc_text[:10000], chunk_text=chunk_text)
    context = llm.generate(prompt, task="context_generation")
    return context

def chunk_text(text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK, generate_context_bool: bool = True) -> List[Dict]:
    """Splits text into sentences and groups them into chunks."""
    sentences = split_into_sentences(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)

    processed_chunks = []
    for chunk in chunks:
        context = ""
        if generate_context_bool:
            context = generate_context(text, chunk)
        processed_chunks.append({"content": chunk, "context": context})
    
    return processed_chunks
