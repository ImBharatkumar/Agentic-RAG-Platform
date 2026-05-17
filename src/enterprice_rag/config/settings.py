import os
from pathlib import Path

GOOGLE_GENAI_USE_VERTEXAI = False


# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_files"
OUTPUT_DIR = BASE_DIR / "output_files"


# ---------- Configuration ----------

DATABASE_URL = "postgresql+psycopg://postgres:Barry#1430@localhost:5432/rag"


# EMBEDDING SETTINGS
# ============================================================================
EMBED_MODEL = "jina-embeddings-v2-base-en"  # Google's embedding API
EMBED_DIM = 768
BATCH_SIZE = 64  # Max 250 for Gemini API

# ============================================================================
# CHUNKING SETTINGS - OPTIMIZED FOR SMALL MODELS
# ============================================================================
SENTENCES_PER_CHUNK = 8  # Reduced from 4 (smaller chunks = more precise retrieval)

# ============================================================================
# LLM SETTINGS - MULTI-MODEL SUPPORT
# ============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Option 1: Multi-model setup (RECOMMENDED)
MODELS = {
    "query_rewrite": "granite4.1:3b",  # Fast query reformulation (Switched from qwen3)
    "reflection": "phi4-mini-reasoning:3.8b",  # Binary decision making
    "generation": "granite4.1:3b",  # Best reasoning for answers
    "context_generation": "phi4-mini-reasoning:3.8b",  # For Contextual Retrieval
    "llm": "granite4.1:3b",  # Fallback/default (Switched from qwen3)
}


# ============================================================================
# RAG SETTINGS
# ============================================================================
DEFAULT_TOP_K = 7  # Reduced from 5 (less noise)
MAX_CONTEXT_LENGTH = 2000  # Max chars per chunk in context
MAX_ITERATIONS = 2  # For agentic loop

# ============================================================================
# API KEYS (for embedding)
# ============================================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB2RAGTPsW3M7LzLMpgVVVGgZgTn5FBI1g")
# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "rag_system.log"

# Create necessary directories
OUTPUT_DIR.mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)
