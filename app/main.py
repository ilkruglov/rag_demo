"""FastAPI entrypoint exposing chatbot, profiles, and document management endpoints."""
import asyncio
import logging
import re
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_profile_catalog, get_raw_documents_dir, get_settings, normalize_profile_id
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    ProfileInfo,
    ProfileListResponse,
)
from app.services.chat import answer, refresh_query_engine_cache
from scripts.ingest import SUPPORTED_EXTENSIONS, clear_indexes, ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
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

_PROFILE_INGESTION_LOCKS: dict[str, asyncio.Lock] = {}
_PROFILE_STATE_CONDITIONS: dict[str, asyncio.Condition] = {}
_ACTIVE_CHAT_REQUESTS: dict[str, int] = {}
_INGESTION_IN_PROGRESS: dict[str, bool] = {}
_DOCUMENT_OP_LOCK_TIMEOUT_SECONDS = 600.0


def _resolve_profile(profile_id: str | None) -> tuple[str, Path]:
    try:
        settings = get_settings(profile_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown profile: {normalize_profile_id(profile_id)}",
        ) from exc

    raw_dir = get_raw_documents_dir(settings)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return settings.active_profile, raw_dir


def _get_ingestion_lock(profile_id: str) -> asyncio.Lock:
    lock = _PROFILE_INGESTION_LOCKS.get(profile_id)
    if lock is None:
        lock = asyncio.Lock()
        _PROFILE_INGESTION_LOCKS[profile_id] = lock
    return lock


def _get_state_condition(profile_id: str) -> asyncio.Condition:
    condition = _PROFILE_STATE_CONDITIONS.get(profile_id)
    if condition is None:
        condition = asyncio.Condition()
        _PROFILE_STATE_CONDITIONS[profile_id] = condition
    return condition


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return sanitized or "uploaded_file"


def _next_available_path(raw_dir: Path, sanitized_name: str) -> Path:
    base = Path(sanitized_name).stem or "uploaded_file"
    suffix = Path(sanitized_name).suffix
    candidate = raw_dir / f"{base}{suffix}"
    index = 1
    while candidate.exists():
        candidate = raw_dir / f"{base}-{index}{suffix}"
        index += 1
    return candidate


def _save_upload(raw_dir: Path, file: UploadFile) -> Path:
    sanitized_name = _sanitize_filename(file.filename or "document")
    destination = _next_available_path(raw_dir, sanitized_name)
    file.file.seek(0)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination


def _list_supported_documents(raw_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in raw_dir.glob("*")
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


async def _wait_for_ingestion_slot(profile_id: str, condition: asyncio.Condition) -> None:
    async with condition:
        while _ACTIVE_CHAT_REQUESTS.get(profile_id, 0) > 0 or _INGESTION_IN_PROGRESS.get(profile_id, False):
            await condition.wait()
        _INGESTION_IN_PROGRESS[profile_id] = True


async def _finish_ingestion_slot(profile_id: str, condition: asyncio.Condition) -> None:
    async with condition:
        _INGESTION_IN_PROGRESS[profile_id] = False
        condition.notify_all()


async def _acquire_document_operation_lock(lock: asyncio.Lock) -> None:
    await asyncio.wait_for(lock.acquire(), timeout=_DOCUMENT_OP_LOCK_TIMEOUT_SECONDS)


def _reindex_or_clear_documents(profile_id: str, raw_dir: Path) -> tuple[bool, bool]:
    if _list_supported_documents(raw_dir):
        ingest(source=raw_dir, profile_id=profile_id)
        refresh_query_engine_cache()
        return True, False

    clear_indexes(profile_id=profile_id)
    refresh_query_engine_cache()
    return False, True


@app.get("/profiles", response_model=ProfileListResponse)
async def list_profiles() -> ProfileListResponse:
    """Return available task profiles and their document counts."""

    catalog = get_profile_catalog()
    profiles: list[ProfileInfo] = []
    for entry in catalog.values():
        raw_dir = Path(entry["raw_documents_dir"])
        total_documents = len(_list_supported_documents(raw_dir)) if raw_dir.exists() else 0
        profiles.append(
            ProfileInfo(
                profile_id=entry["profile_id"],
                label=entry["label"],
                description=entry["description"],
                is_active=entry["is_active"],
                qdrant_path=entry["qdrant_path"],
                storage_dir=entry["storage_dir"],
                raw_documents_dir=entry["raw_documents_dir"],
                qdrant_collection=entry["qdrant_collection"],
                total_documents=total_documents,
            )
        )

    profiles.sort(key=lambda item: (not item.is_active, item.label.lower()))
    return ProfileListResponse(
        active_profile=get_settings().active_profile,
        profiles=profiles,
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(profile_id: str | None = Query(default=None)) -> DocumentListResponse:
    """List documents currently included in the selected knowledge base."""

    resolved_profile, raw_dir = _resolve_profile(profile_id)
    docs = _list_supported_documents(raw_dir)
    return DocumentListResponse(
        documents=[_build_document_info(path) for path in docs],
        total_documents=len(docs),
        profile_id=resolved_profile,
    )


@app.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    profile_id: str | None = Query(default=None),
    files: List[UploadFile] = File(...),
) -> DocumentUploadResponse:
    """Upload documents and rebuild the selected vector index."""

    resolved_profile, raw_dir = _resolve_profile(profile_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for upload in files:
        extension = Path(upload.filename or "").suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension or 'unknown'}")

    lock = _get_ingestion_lock(resolved_profile)
    condition = _get_state_condition(resolved_profile)
    saved_files: list[str] = []
    lock_acquired = False
    try:
        await _acquire_document_operation_lock(lock)
        lock_acquired = True

        for upload in files:
            path = _save_upload(raw_dir, upload)
            saved_files.append(path.name)

        await _wait_for_ingestion_slot(resolved_profile, condition)
        try:
            await run_in_threadpool(partial(_reindex_or_clear_documents, resolved_profile, raw_dir))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Document ingestion failed for profile %s: %s", resolved_profile, exc)
            raise HTTPException(status_code=500, detail="Document ingestion failed") from exc
        finally:
            await _finish_ingestion_slot(resolved_profile, condition)
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
        total_documents=len(_list_supported_documents(raw_dir)),
        profile_id=resolved_profile,
    )


@app.post("/documents/add", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def add_documents(
    profile_id: str | None = Query(default=None),
    files: List[UploadFile] = File(...),
) -> DocumentUploadResponse:
    """Alias for /documents to simplify demo integrations."""

    return await upload_documents(profile_id=profile_id, files=files)


@app.post("/documents/delete", response_model=DocumentDeleteResponse)
async def delete_documents(payload: DocumentDeleteRequest) -> DocumentDeleteResponse:
    """Delete documents from a profile KB and rebuild or clear its index."""

    resolved_profile, raw_dir = _resolve_profile(payload.profile_id)

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
    missing = [name for name in requested if not (raw_dir / name).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(missing)}")

    lock = _get_ingestion_lock(resolved_profile)
    condition = _get_state_condition(resolved_profile)
    lock_acquired = False

    try:
        await _acquire_document_operation_lock(lock)
        lock_acquired = True

        for file_name in requested:
            (raw_dir / file_name).unlink(missing_ok=False)

        await _wait_for_ingestion_slot(resolved_profile, condition)
        try:
            reindexed, indexes_cleared = await run_in_threadpool(
                partial(_reindex_or_clear_documents, resolved_profile, raw_dir)
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Document deletion reindex failed for profile %s: %s", resolved_profile, exc)
            raise HTTPException(status_code=500, detail="Document deletion failed") from exc
        finally:
            await _finish_ingestion_slot(resolved_profile, condition)
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
        total_documents=len(_list_supported_documents(raw_dir)),
        profile_id=resolved_profile,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Run the RAG pipeline for the selected profile."""

    resolved_profile, _ = _resolve_profile(payload.profile_id)
    condition = _get_state_condition(resolved_profile)

    async with condition:
        if _INGESTION_IN_PROGRESS.get(resolved_profile, False):
            raise HTTPException(status_code=503, detail="Document ingestion in progress")
        _ACTIVE_CHAT_REQUESTS[resolved_profile] = _ACTIVE_CHAT_REQUESTS.get(resolved_profile, 0) + 1

    result: dict[str, Any] = {}
    try:
        result = await run_in_threadpool(answer, payload.question, resolved_profile)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Vector store is not initialized") from error
    finally:
        async with condition:
            _ACTIVE_CHAT_REQUESTS[resolved_profile] = max(
                0,
                _ACTIVE_CHAT_REQUESTS.get(resolved_profile, 1) - 1,
            )
            if _ACTIVE_CHAT_REQUESTS[resolved_profile] == 0:
                condition.notify_all()

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        profile_id=resolved_profile,
    )
