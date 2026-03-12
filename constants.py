import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", OLLAMA_MODEL)
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))

TEXT_SPLITTER_CHUNK_SIZE = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE", "500"))
TEXT_SPLITTER_CHUNK_OVERLAP = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP", "50"))

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents/uploads")

# Speed/latency controls
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "0").strip().lower() in {"1", "true", "yes", "y"}

RETRIEVER_K_DENSE = int(os.getenv("RETRIEVER_K_DENSE", "2"))
RETRIEVER_K_SPARSE = int(os.getenv("RETRIEVER_K_SPARSE", "4"))

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "4000"))
