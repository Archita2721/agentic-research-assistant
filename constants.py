import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", OLLAMA_MODEL)

TEXT_SPLITTER_CHUNK_SIZE = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE", "500"))
TEXT_SPLITTER_CHUNK_OVERLAP = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP", "50"))

