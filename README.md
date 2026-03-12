# Agentic Research Assistant

An **Agentic AI system** that performs research over documents using RAG and agent workflows.

This project demonstrates how to build an AI agent backend using **LangGraph**, **FastAPI**, and **Ollama**.

The system allows users to upload documents, store them in a vector database, and query them using an AI-powered research agent.

## Features

* Document upload and indexing
* Retrieval-Augmented Generation (RAG)
* Agent workflows using LangGraph
* Local LLM inference using Ollama
* Vector search using FAISS
* FastAPI backend

## Architecture

User Query -> FastAPI API -> LangGraph Agent Workflow -> Retriever (FAISS Vector DB) -> Ollama LLM -> Final Answer

---

## Tech Stack

* Python
* FastAPI
* LangGraph
* LangChain
* Ollama
* FAISS
* Pydantic

## Project Structure

```
agentic-research-assistant/
│
├─ main.py
├─ requirements.txt
│
├─ agents/
│   ├─ planner.py
│   └─ writer.py
│
├─ tools/
│   ├─ search_tool.py
│   ├─ rag_tool.py
│   ├─ document_loader.py
│   └─ text_splitter.py
│
├─ vectorstore/
│   └─ faiss_store.py
│
└─ app/
    ├─ graph.py
    └─ state.py
```

## Installation

Clone the repository:

```
git clone https://github.com/Archita2721/agentic-research-assistant.git
```

Create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn main:app --reload
```

Open API docs in your browser:

```
http://127.0.0.1:8000/docs
```

## API Endpoints

### Upload Document

```
POST /upload
```

Upload a PDF file to index it in the vector database.

### Ask Question

```
POST /ask
```

Example request:

```json
```json
{
  "question": "Summarize the document"
  "question": "Summarize the document"
}
```

## Future Improvements

* Multi-document knowledge base
* Tool routing agents
* Self-reflection agents
* Streaming responses
* Conversation memory
* Research report generation

## Goal

This project explores **Agentic AI architectures** and demonstrates how to build intelligent research assistants using modern LLM frameworks.

