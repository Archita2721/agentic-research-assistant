import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse

from constants import DOCUMENTS_DIR
from app.enums import SSEEvent, Step
from app.graph import build_graph
from app.schemas import AskData, Query, UploadData
from app.utils import api_error, api_ok, sanitize_filename, sse
from agents.router import router_agent
from agents.planner import planner_agent
from agents.critic import critic_agent
from agents.writer import build_writer_prompt
from agents.smalltalk import build_smalltalk_prompt
from tools.document_loader import load_document
from tools.search_tool import search_agent
from tools.text_splitter import split_documents
from tools.rag_tool import rag_agent
from vectorstore.faiss_store import add_documents_dense, add_documents_sparse
from constants import ENABLE_WEB_SEARCH

router = APIRouter()
graph = build_graph()

_uploads_dir = Path(DOCUMENTS_DIR)
_uploads_dir.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        total_start = time.perf_counter()
        doc_id = uuid4().hex
        original_filename = file.filename or "upload"
        safe_name = sanitize_filename(original_filename)
        stored_filename = f"{doc_id}_{safe_name}"
        file_path = _uploads_dir / stored_filename

        save_start = time.perf_counter()
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[timing] upload.save_file {(time.perf_counter() - save_start):.3f}s stored={stored_filename}", flush=True)

        load_split_start = time.perf_counter()
        docs = load_document(
            str(file_path),
            doc_id=doc_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            content_type=file.content_type,
        )
        chunks = split_documents(docs)
        print(
            f"[timing] upload.load_and_split {(time.perf_counter() - load_split_start):.3f}s docs={len(docs)} chunks={len(chunks)}",
            flush=True,
        )

        sparse_start = time.perf_counter()
        add_documents_sparse(chunks)
        print(f"[timing] upload.index_sparse {(time.perf_counter() - sparse_start):.3f}s", flush=True)
        background_tasks.add_task(add_documents_dense, chunks)

        print(f"[timing] upload.total {(time.perf_counter() - total_start):.3f}s", flush=True)
        payload = UploadData(doc_id=doc_id, stored_filename=stored_filename, dense_indexing="queued").model_dump()
        return api_ok(message="Document uploaded and indexed successfully", data=payload)
    except ValueError as exc:
        return api_error(code="unsupported_file", message=str(exc), status_code=400)
    except Exception as exc:
        return api_error(code="upload_failed", message=str(exc), status_code=500)


@router.post("/ask")
def ask_question(query: Query):
    try:
        start = time.perf_counter()
        result = graph.invoke({"question": query.question})
        print(f"[timing] ask.total {(time.perf_counter() - start):.3f}s", flush=True)
        data = AskData(
            answer=result.get("final_answer", ""),
            critique=result.get("critique"),
            intent=result.get("intent"),
        ).model_dump()
        return api_ok(data=data)
    except Exception as exc:
        return api_error(code="ask_failed", message=str(exc), status_code=500)


@router.post("/ask/stream")
def ask_question_stream(query: Query):
    def generate():
        total_start = time.perf_counter()
        state: dict = {"question": query.question, "search_results": [], "documents": []}

        yield sse(SSEEvent.STATUS.value, {"step": Step.ROUTER.value, "message": "Routing request"})
        state.update(router_agent(state))

        if state.get("route") == "smalltalk":
            yield sse(SSEEvent.STATUS.value, {"step": Step.SMALLTALK.value, "message": "Generating a quick reply"})
            st_start = time.perf_counter()
            # Stream smalltalk tokens too
            from llm import chat_llm  # local import

            prompt = build_smalltalk_prompt(state.get("question", ""))

            reply = ""
            for chunk in chat_llm.stream(prompt):
                token = getattr(chunk, "text", "") or ""
                if not token:
                    continue
                reply += token
                yield sse(SSEEvent.TOKEN.value, {"text": token})

            state["final_answer"] = reply.strip()
            yield sse(SSEEvent.TIMING.value, {"step": Step.SMALLTALK.value, "elapsed_s": round(time.perf_counter() - st_start, 3)})
            yield sse(SSEEvent.FINAL.value, api_ok(data=AskData(answer=state.get("final_answer", ""), intent=state.get("intent")).model_dump()))
            yield sse(SSEEvent.DONE.value, {"elapsed_s": round(time.perf_counter() - total_start, 3)})
            return

        if ENABLE_WEB_SEARCH:
            yield sse(SSEEvent.STATUS.value, {"step": Step.PLANNER.value, "message": "Planning web search"})
            planner_start = time.perf_counter()
            state.update(planner_agent(state))
            yield sse(SSEEvent.TIMING.value, {"step": Step.PLANNER.value, "elapsed_s": round(time.perf_counter() - planner_start, 3)})

            yield sse(SSEEvent.STATUS.value, {"step": Step.WEB_SEARCH.value, "message": "Searching the web"})
            search_start = time.perf_counter()
            state.update(search_agent(state))
            yield sse(SSEEvent.TIMING.value, {"step": Step.WEB_SEARCH.value, "elapsed_s": round(time.perf_counter() - search_start, 3)})

        yield sse(SSEEvent.STATUS.value, {"step": Step.RETRIEVE.value, "message": "Retrieving from knowledge base"})
        retrieve_start = time.perf_counter()
        state.update(rag_agent(state))
        yield sse(SSEEvent.TIMING.value, {"step": Step.RETRIEVE.value, "elapsed_s": round(time.perf_counter() - retrieve_start, 3)})

        # Draft answer (token streaming)
        yield sse(SSEEvent.STATUS.value, {"step": Step.WRITER.value, "message": "Generating draft answer"})
        writer_prompt = build_writer_prompt(
            state.get("question", ""),
            state.get("documents", []),
            state.get("search_results", []),
        )

        from llm import chat_llm  # local import to avoid import-time side effects

        draft_start = time.perf_counter()
        draft = ""
        for chunk in chat_llm.stream(writer_prompt):
            token = getattr(chunk, "text", "") or ""
            if not token:
                continue
            draft += token
            yield sse(SSEEvent.TOKEN.value, {"text": token})
        yield sse(SSEEvent.TIMING.value, {"step": Step.WRITER.value, "elapsed_s": round(time.perf_counter() - draft_start, 3)})

        state["final_answer"] = draft.strip()

        yield sse(SSEEvent.STATUS.value, {"step": Step.CRITIC.value, "message": "Self-critic reviewing answer"})
        critic_start = time.perf_counter()
        state.update(critic_agent(state))
        yield sse(SSEEvent.TIMING.value, {"step": Step.CRITIC.value, "elapsed_s": round(time.perf_counter() - critic_start, 3)})

        yield sse(
            SSEEvent.FINAL.value,
            api_ok(
                data=AskData(
                    answer=state.get("final_answer", ""),
                    critique=state.get("critique", ""),
                    intent=state.get("intent"),
                ).model_dump()
            ),
        )
        yield sse(SSEEvent.DONE.value, {"elapsed_s": round(time.perf_counter() - total_start, 3)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
