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
API_VERSION = os.getenv("API_VERSION", "1")

# MongoDB (local via Compass)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?appName=Agentic_Project")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "agentic_research")
MONGODB_UPLOADS_COLLECTION = os.getenv("MONGODB_UPLOADS_COLLECTION", "uploads")
MONGODB_CHUNKS_COLLECTION = os.getenv("MONGODB_CHUNKS_COLLECTION", "chunks")
MONGODB_JOBS_COLLECTION = os.getenv("MONGODB_JOBS_COLLECTION", "job")
MONGODB_MESSAGES_COLLECTION = os.getenv("MONGODB_MESSAGES_COLLECTION", "messages")

# Conversational memory
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "12"))

# Local persistence (to survive server restarts)
FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "vectorstore_data/faiss")
BOOTSTRAP_SPARSE_FROM_MONGO = os.getenv("BOOTSTRAP_SPARSE_FROM_MONGO", "1").strip().lower() in {"1", "true", "yes", "y"}
BOOTSTRAP_DENSE_FROM_DISK = os.getenv("BOOTSTRAP_DENSE_FROM_DISK", "1").strip().lower() in {"1", "true", "yes", "y"}

# Speed/latency controls
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "0").strip().lower() in {"1", "true", "yes", "y"}

RETRIEVER_K_DENSE = int(os.getenv("RETRIEVER_K_DENSE", "2"))
RETRIEVER_K_SPARSE = int(os.getenv("RETRIEVER_K_SPARSE", "4"))

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "4000"))

# Intent / smalltalk detection
SMALLTALK_GREETING_FIRST = {"hi", "hello", "hey", "yo", "sup", "wassup"}
SMALLTALK_THANKS_FIRST = {"thanks", "thank", "thx"}
SMALLTALK_GREETING_PHRASES = {"whats up", "what's up", "thank you"}

# Memory intent phrases (conversation history questions)
MEMORY_INTENT_PHRASES = {
    "what did i ask earlier",
    "what did i ask before",
    "what did i say earlier",
    "what did i say before",
    "show my previous question",
    "show my previous questions",
    "show conversation",
    "show conversation history",
    "show chat history",
    "chat history",
    "conversation history",
}

# Retrieval / summarization heuristics
GENERIC_DOC_QUESTION_PHRASES = {
    "summarize",
    "summary",
    "summarize the document",
    "summarise the document",
    "tell me about the document",
    "tell me about document",
    "what is mentioned",
    "what is mentioned in the document",
    "what does the document say",
    "what's in the document",
    "whats in the document",
    "overview",
    "give an overview",
}

QUESTION_KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "do",
    "does",
    "give",
    "hi",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "thanks",
    "that",
    "the",
    "this",
    "to",
    "what",
    "whats",
    "what's",
    "you",
    "your",
}
