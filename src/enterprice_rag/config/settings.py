import os
from pathlib import Path
from dotenv import load_dotenv

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
# Load .env from root directory (parent of src)
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")

GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1", "yes")

# Directories
DATA_DIR = BASE_DIR / "data_files"
OUTPUT_DIR = BASE_DIR / "output_files"

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Embedding
EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

# Chunking
SENTENCES_PER_CHUNK = int(os.getenv("SENTENCES_PER_CHUNK"))

# Ollama LLM Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

MODELS = {
    "query_rewrite": os.getenv("MODEL_QUERY_REWRITE"),
    "reflection": os.getenv("MODEL_REFLECTION"),
    "generation": os.getenv("MODEL_GENERATION"),
    "context_generation": os.getenv("MODEL_CONTEXT_GENERATION"),
    "llm": os.getenv("MODEL_DEFAULT"),
}

# RAG Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS"))

# API Keys
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL")
LOG_FILE = BASE_DIR / "logs" / "rag_system.log"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)
