from langchain_ollama import ChatOllama, OllamaEmbeddings

from constants import OLLAMA_EMBEDDINGS_MODEL, OLLAMA_MODEL, OLLAMA_NUM_PREDICT, OLLAMA_TEMPERATURE

chat_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=OLLAMA_NUM_PREDICT,
)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
