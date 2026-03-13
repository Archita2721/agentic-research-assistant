# Agentic Research Assistant

An agentic AI system that answers questions over your uploaded files using RAG + agent workflows (FastAPI + LangGraph + Ollama).

## How It Works

1. **Upload**: You upload a document to `POST /upload`. The file is saved under `documents/` with a UUID prefix for uniqueness.
2. **Load + Split**: The server loads the file (PDF/TXT/MD/CSV/Excel/DOCX) and splits it into chunks. Each chunk includes metadata like `doc_id`, filenames, and `chunk_id`.
3. **Index (Multi-document KB)**:
   - **BM25 (fast)** is built immediately so questions can be answered right away.
   - **FAISS (dense embeddings)** is built/updated in the background and merged when ready.
4. **Ask**: You call `POST /ask` (or stream via `POST /ask/stream`).
5. **Intent Routing**:
   - **Smalltalk** (hi/thanks/wassup) bypasses retrieval and responds directly via the LLM with suggestions.
   - **Research** queries run retrieval (hybrid BM25 + FAISS when available).
6. **Answer + Self-Critic**: The writer drafts an answer from the retrieved context, then a self-critic agent reviews it and outputs the final answer.

## Features

* Multi-document knowledge base (incremental indexing)
* Hybrid retrieval (BM25 + FAISS when ready)
* Intent routing (smalltalk bypasses RAG)
* Streaming answers via SSE (`/ask/stream`)

## Supported Upload Types

`pdf`, `txt`, `md/markdown`, `csv`, `xls`, `xlsx`, `docx`

Uploads are stored under `documents/` (ignored by git).

## Quickstart

Create `.env` from `.env.example`, then:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open docs:

```
http://127.0.0.1:8000/docs
```

## Environment Variables

See `.env.example` for the full list. Common ones:

* `OLLAMA_MODEL`, `OLLAMA_EMBEDDINGS_MODEL`
* `TEXT_SPLITTER_CHUNK_SIZE`, `TEXT_SPLITTER_CHUNK_OVERLAP`
* `DOCUMENTS_DIR`
* `MONGODB_URI`, `MONGODB_DB_NAME`
* `ENABLE_WEB_SEARCH`

## API

### `POST /upload`

Uploads a file and indexes it. The server uses a **cookie-based session** and will set a `session_id` cookie if one does not exist yet.

Notes:
- The request does **not** accept `session_id` (it comes only from the cookie).
- The response does **not** expose `doc_id` (document selection is internal and session-scoped).

### `POST /ask`

Request:

```json
{ "question": "Summarize the document" }
```

Session behavior:
- The server reads `session_id` from `Cookie: session_id=...`.
- If missing, the server generates one and sets `Set-Cookie: session_id=...`.
- Browsers (Swagger UI) send cookies automatically on subsequent requests.
- For `curl`, use a cookie jar so the session persists:

```bash
curl -c cookies.txt -b cookies.txt -F "file=@documents/example.pdf" http://127.0.0.1:8000/upload
curl -c cookies.txt -b cookies.txt -H "Content-Type: application/json" -d "{\"question\":\"Summarize the document\"}" http://127.0.0.1:8000/ask
```

Document selection:
- The app automatically uses the **latest uploaded document for the current session**.
- It never falls back to a "latest document overall" (prevents cross-session data leakage).

How routing works:

- **Memory questions** (e.g. "what did I ask earlier") are answered from MongoDB chat history and **do not** run RAG.
- **Smalltalk** (hi/thanks/wassup) bypasses RAG and replies directly via the LLM with suggestions.
- **Research** questions run retrieval (BM25 + FAISS when available) and then writer + self-critic.

All JSON endpoints follow the same response shape:

```json
{
  "ok": true,
  "message": null,
  "data": { "answer": "...", "critique": "...", "intent": "research" },
  "error": null,
  "meta": null
}
```

### `POST /ask/stream` (SSE)

Streams `text/event-stream` events:

* `status` (`router`, `smalltalk`, `memory`, `planner`, `web_search`, `retrieve`, `writer`, `critic`)
* `timing`
* `token`
* `final` (same shape as `POST /ask` via `api_ok(...)`)
* `done`

## Project Structure

```
main.py
constants.py
llm.py

app/
agents/
tools/
vectorstore/
documents/
```
