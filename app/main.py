"""FastAPI entrypoint exposing chatbot and document management endpoints."""
import asyncio
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_raw_documents_dir, get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.chat import answer, refresh_query_engine_cache
from scripts.ingest import SUPPORTED_EXTENSIONS, clear_indexes, ingest

# Configure logging for app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
# Set specific module loggers to INFO
logging.getLogger("app.services.chat").setLevel(logging.INFO)
logging.getLogger("app.services.bm25_retriever").setLevel(logging.INFO)

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Universal RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = get_settings()
_RAW_DATA_DIR = get_raw_documents_dir(_settings)
_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

_INGESTION_LOCK: asyncio.Lock | None = None
_STATE_CONDITION: asyncio.Condition | None = None
_ACTIVE_CHAT_REQUESTS = 0
_INGESTION_IN_PROGRESS = False
_DOCUMENT_OP_LOCK_TIMEOUT_SECONDS = 600.0


def _get_ingestion_lock() -> asyncio.Lock:
    global _INGESTION_LOCK
    if _INGESTION_LOCK is None:
        _INGESTION_LOCK = asyncio.Lock()
    return _INGESTION_LOCK


def _get_state_condition() -> asyncio.Condition:
    global _STATE_CONDITION
    if _STATE_CONDITION is None:
        _STATE_CONDITION = asyncio.Condition()
    return _STATE_CONDITION


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return sanitized or "uploaded_file"


def _next_available_path(sanitized_name: str) -> Path:
    base = Path(sanitized_name).stem or "uploaded_file"
    suffix = Path(sanitized_name).suffix
    candidate = _RAW_DATA_DIR / f"{base}{suffix}"
    index = 1
    while candidate.exists():
        candidate = _RAW_DATA_DIR / f"{base}-{index}{suffix}"
        index += 1
    return candidate


def _save_upload(file: UploadFile) -> Path:
    sanitized_name = _sanitize_filename(file.filename or "document")
    destination = _next_available_path(sanitized_name)
    file.file.seek(0)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination


def _list_supported_documents() -> list[Path]:
    return sorted(
        [
            path
            for path in _RAW_DATA_DIR.glob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def _build_document_info(path: Path) -> DocumentInfo:
    stat_result = path.stat()
    return DocumentInfo(
        file_name=path.name,
        size_bytes=stat_result.st_size,
        modified_at=datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
    )


async def _wait_for_ingestion_slot(condition: asyncio.Condition) -> None:
    global _INGESTION_IN_PROGRESS, _ACTIVE_CHAT_REQUESTS
    async with condition:
        while _ACTIVE_CHAT_REQUESTS > 0 or _INGESTION_IN_PROGRESS:
            await condition.wait()
        _INGESTION_IN_PROGRESS = True


async def _finish_ingestion_slot(condition: asyncio.Condition) -> None:
    global _INGESTION_IN_PROGRESS
    async with condition:
        _INGESTION_IN_PROGRESS = False
        condition.notify_all()


async def _acquire_document_operation_lock(lock: asyncio.Lock) -> None:
    """Wait for an exclusive documents operation slot."""

    await asyncio.wait_for(lock.acquire(), timeout=_DOCUMENT_OP_LOCK_TIMEOUT_SECONDS)


def _reindex_or_clear_documents() -> tuple[bool, bool]:
    if _list_supported_documents():
        ingest()
        refresh_query_engine_cache()
        return True, False

    clear_indexes()
    refresh_query_engine_cache()
    return False, True


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List documents currently included in the knowledge base."""

    docs = _list_supported_documents()
    return DocumentListResponse(
        documents=[_build_document_info(path) for path in docs],
        total_documents=len(docs),
    )


@app.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    """Upload documents and rebuild the vector index."""

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for upload in files:
        extension = Path(upload.filename or "").suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension or 'unknown'}")

    lock = _get_ingestion_lock()
    condition = _get_state_condition()
    saved_files: list[str] = []
    lock_acquired = False
    try:
        await _acquire_document_operation_lock(lock)
        lock_acquired = True
    
        for upload in files:
            path = _save_upload(upload)
            saved_files.append(path.name)

        await _wait_for_ingestion_slot(condition)
        try:
            await run_in_threadpool(_reindex_or_clear_documents)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Document ingestion failed: %s", exc)
            raise HTTPException(status_code=500, detail="Document ingestion failed") from exc
        finally:
            await _finish_ingestion_slot(condition)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=409,
            detail="Another document operation is still in progress. Please retry shortly.",
        ) from None
    finally:
        if lock_acquired:
            lock.release()

    return DocumentUploadResponse(
        saved_files=saved_files,
        ingestion_started=True,
        total_documents=len(_list_supported_documents()),
    )


@app.post("/documents/add", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def add_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    """Alias for /documents to simplify demo integrations."""

    return await upload_documents(files=files)


@app.post("/documents/delete", response_model=DocumentDeleteResponse)
async def delete_documents(payload: DocumentDeleteRequest) -> DocumentDeleteResponse:
    """Delete documents from KB and rebuild (or clear) vector index."""

    if not payload.file_names:
        raise HTTPException(status_code=400, detail="No files provided")

    sanitized_names: list[str] = []
    for file_name in payload.file_names:
        clean_name = Path(file_name).name
        if clean_name != file_name:
            raise HTTPException(status_code=400, detail=f"Invalid file name: {file_name}")
        if Path(clean_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_name}")
        sanitized_names.append(clean_name)

    requested = sorted(set(sanitized_names))
    missing = [name for name in requested if not (_RAW_DATA_DIR / name).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(missing)}")

    lock = _get_ingestion_lock()
    condition = _get_state_condition()
    lock_acquired = False

    try:
        await _acquire_document_operation_lock(lock)
        lock_acquired = True

        for file_name in requested:
            (_RAW_DATA_DIR / file_name).unlink(missing_ok=False)

        await _wait_for_ingestion_slot(condition)
        try:
            reindexed, indexes_cleared = await run_in_threadpool(_reindex_or_clear_documents)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Document deletion reindex failed: %s", exc)
            raise HTTPException(status_code=500, detail="Document deletion failed") from exc
        finally:
            await _finish_ingestion_slot(condition)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=409,
            detail="Another document operation is still in progress. Please retry shortly.",
        ) from None
    finally:
        if lock_acquired:
            lock.release()

    return DocumentDeleteResponse(
        deleted_files=requested,
        reindexed=reindexed,
        indexes_cleared=indexes_cleared,
        total_documents=len(_list_supported_documents()),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Chat endpoint that runs the RAG pipeline."""

    condition = _get_state_condition()
    global _ACTIVE_CHAT_REQUESTS

    async with condition:
        if _INGESTION_IN_PROGRESS:
            raise HTTPException(status_code=503, detail="Document ingestion in progress")
        _ACTIVE_CHAT_REQUESTS += 1

    result: dict[str, Any] = {}
    try:
        result = await run_in_threadpool(answer, payload.question)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Vector store is not initialized") from error
    finally:
        async with condition:
            _ACTIVE_CHAT_REQUESTS -= 1
            if _ACTIVE_CHAT_REQUESTS == 0:
                condition.notify_all()

    return ChatResponse(answer=result["answer"], sources=result["sources"])
