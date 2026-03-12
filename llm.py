from langchain_ollama import ChatOllama, OllamaEmbeddings

from constants import OLLAMA_EMBEDDINGS_MODEL, OLLAMA_MODEL

chat_llm = ChatOllama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
