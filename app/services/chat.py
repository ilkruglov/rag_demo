"""Retrieval and response pipeline using LlamaIndex and Qdrant."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from llama_index.core import Settings as LlamaSettings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES, QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.config import DEFAULT_STOP_SEQUENCES, get_settings
from app.services.index_names import VECTOR_INDEX_ID
from app.services.reranker import KeywordOverlapReranker, OperationCodeBooster, SectionTitleBooster, SemanticDirectionBooster
from app.services.bm25_retriever import BM25Retriever, get_bm25_index, clear_bm25_cache
from app.services.semantic_enrichment import expand_query_with_codes

try:
    from llama_index.core.postprocessor import SentenceTransformerRerank as _SentenceTransformerRerank
except ImportError:  # pragma: no cover - optional dependency
    _SentenceTransformerRerank = None

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class MetadataStripper(BaseNodePostprocessor):
    """Remove technical metadata fields before sending context to LLM."""

    keys_to_remove: list[str] = ["chunk_id", "source", "document_filename"]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        for node in nodes:
            if node.node.metadata:
                for key in self.keys_to_remove:
                    node.node.metadata.pop(key, None)
        return nodes




LOGGER = logging.getLogger(__name__)


_QDRANT_CLIENT: QdrantClient | None = None
_QDRANT_CLIENT_PATH: str | None = None
_QDRANT_ASYNC_CLIENT: Any | None = None
_QDRANT_ASYNC_CLIENT_PATH: str | None = None


def _strip_internal_thoughts(text: str) -> str:
    sanitized = text
    while "<think>" in sanitized:
        start = sanitized.find("<think>")
        end = sanitized.find("</think>", start)
        if end == -1:
            sanitized = sanitized.replace("<think>", "")
            break
        sanitized = sanitized[:start] + sanitized[end + len("</think>"):]
    sanitized = sanitized.replace("</think>", "")
    return sanitized.strip()


# Markers that indicate unwanted "extra info" sections to strip
_UNWANTED_SECTION_MARKERS = [
    "\n**Дополнительно:**",
    "\nДополнительно:",
    "\n**Важно:**",
    "\nВажно:",
    "\n**Примечание:**",
    "\nПримечание:",
    "\n\n**Дополнительно:**",
    "\n\n**Важно:**",
    "\n\n**Примечание:**",
]


def _strip_unwanted_sections(text: str) -> str:
    """Remove unwanted 'notes/important/additional' sections from the end of the response."""
    result = text
    for marker in _UNWANTED_SECTION_MARKERS:
        pos = result.find(marker)
        if pos != -1:
            # Keep only content before this marker
            result = result[:pos].rstrip()
    return result


def _build_excerpt(text: str, limit: int = 280) -> str | None:
    snippet = " ".join(text.split())
    if not snippet:
        return None
    if len(snippet) <= limit:
        return snippet
    trimmed = snippet[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}…"


def _settings_signature() -> str:
    settings = get_settings()
    fingerprint = {
        "groq_model": settings.groq_model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_output_tokens": settings.max_output_tokens,
        "stop_sequences": settings.stop_sequences,
        "retrieval_top_k": settings.retrieval_top_k,
        "system_prompt": settings.system_prompt,
        "qdrant_path": settings.qdrant_path,
        "qdrant_collection": settings.qdrant_collection,
        "storage_dir": settings.storage_dir,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "reranker_top_n": settings.reranker_top_n,
        "domain_rules_enabled": settings.domain_rules_enabled,
    }
    return json.dumps(fingerprint, sort_keys=True)


@lru_cache(maxsize=1)
def _build_query_engine(signature: str):  # noqa: ARG001
    settings = get_settings()

    if settings.force_offline_mode:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        query_instruction="query: ",
        text_instruction="passage: ",
    )
    LlamaSettings.embed_model = embed_model

    client = _get_qdrant_client(settings.qdrant_path)
    async_client = _get_qdrant_async_client(settings.qdrant_path)
    vector_store = QdrantVectorStore(
        client=client,
        aclient=async_client,
        collection_name=settings.qdrant_collection,
        prefer_grpc=False,
    )

    storage_dir = Path(settings.storage_dir)
    if not storage_dir.exists():
        raise FileNotFoundError(
            "Хранилище индекса не найдено. Выполните `python -m scripts.ingest`, чтобы создать базу знаний."
        )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(storage_dir),
    )
    vector_index = load_index_from_storage(storage_context, index_id=VECTOR_INDEX_ID)

    # Получаем nodes для BM25 из docstore
    docstore = storage_context.docstore
    all_nodes = list(docstore.docs.values())

    llm = Groq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        top_p=settings.top_p,
        stop=settings.stop_sequences or DEFAULT_STOP_SEQUENCES,
        api_base=settings.groq_api_base,
        context_window=32768,  # Qwen3-32B supports 32K context
    )
    LlamaSettings.llm = llm

    vector_retriever = vector_index.as_retriever(similarity_top_k=settings.retrieval_top_k)
    combined_retriever = vector_retriever

    # BM25 retriever с русской лемматизацией
    if all_nodes:
        try:
            bm25_index = get_bm25_index(settings.storage_dir, all_nodes)
            bm25_retriever = BM25Retriever(
                index=bm25_index,
                similarity_top_k=settings.retrieval_top_k,
            )
            combined_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=settings.retrieval_top_k,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                num_queries=1,
                retriever_weights=[0.6, 0.4],  # Vector 60% + BM25 40%
                llm=llm,
            )
            LOGGER.info("Using hybrid retrieval: Vector + BM25 with lemmatization")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to create BM25 retriever, using vector only: %s", exc)

    node_postprocessors = []

    if settings.domain_rules_enabled:
        # Optional domain-specific boosters for operation-heavy corpora.
        node_postprocessors.append(OperationCodeBooster(boost_factor=0.5))
        LOGGER.info("OperationCodeBooster enabled (boost_factor=0.5)")

        node_postprocessors.append(SemanticDirectionBooster(boost_factor=0.6))
        LOGGER.info("SemanticDirectionBooster enabled (boost_factor=0.6)")
    else:
        LOGGER.info("Domain rules disabled: operation boosters are off")

    # Boost nodes where section_title matches query keywords (before reranking)
    node_postprocessors.append(SectionTitleBooster(boost_factor=0.3))
    LOGGER.info("SectionTitleBooster enabled (boost_factor=0.3)")

    if settings.reranker_model and settings.reranker_top_n > 0:
        reranker = _get_reranker(settings.reranker_model, min(settings.reranker_top_n, settings.retrieval_top_k))
        node_postprocessors.append(reranker)
        LOGGER.info("Reranker enabled: %s (top_n=%d)", settings.reranker_model, settings.reranker_top_n)
    else:
        LOGGER.warning("Reranker disabled: model=%s, top_n=%s", settings.reranker_model, settings.reranker_top_n)

    # Remove technical metadata (chunk_id, source, document_filename) before sending to LLM
    node_postprocessors.append(MetadataStripper())

    text_qa_template = PromptTemplate(
        settings.system_prompt + "\n\nКонтекст:\n{context_str}\n\nВопрос:\n{query_str}"
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        text_qa_template=text_qa_template,
    )

    return RetrieverQueryEngine.from_args(
        retriever=combined_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors or None,
    )


def refresh_query_engine_cache() -> None:
    """Clear cached query engine so new settings take effect."""

    _build_query_engine.cache_clear()
    clear_bm25_cache()


@lru_cache(maxsize=1)
def _get_reranker(model_name: str, top_n: int):
    if model_name.lower() == "keyword_overlap":
        return KeywordOverlapReranker(top_n=top_n)

    if _SentenceTransformerRerank is None:
        LOGGER.warning(
            "sentence-transformers stack is not available; falling back to keyword reranker",
        )
        return KeywordOverlapReranker(top_n=top_n)

    try:
        return _SentenceTransformerRerank(model=model_name, top_n=top_n)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Falling back to keyword reranker after failing to load %s: %s",
            model_name,
            exc,
        )
        return KeywordOverlapReranker(top_n=top_n)


def answer(question: str) -> dict[str, Any]:
    signature = _settings_signature()
    query_engine = _build_query_engine(signature)
    settings = get_settings()

    expanded_question = question
    if settings.domain_rules_enabled:
        expanded_question = expand_query_with_codes(question)
        if expanded_question != question:
            LOGGER.info("Query expanded: %s -> %s", question[:50], expanded_question[:80])

    response = query_engine.query(expanded_question)
    raw_answer = getattr(response, "response", str(response))
    cleaned_answer = _strip_internal_thoughts(raw_answer)
    cleaned_answer = _strip_unwanted_sections(cleaned_answer)

    # Не добавляем **Ответ:** если модель уже его добавила
    if cleaned_answer and not cleaned_answer.lstrip().startswith("**Ответ:**"):
        cleaned_answer = f"**Ответ:** {cleaned_answer}"
    for stale_phrase in (
        "В предоставленных документах нет данных",
        "В предоставленных документах не содержится",
        "В предоставленных документах не указаны",
    ):
        if stale_phrase in cleaned_answer:
            cleaned_answer = cleaned_answer.replace(stale_phrase, "Я не знаю")
    if not cleaned_answer:
        cleaned_answer = "**Ответ:** Я не знаю."

    sources = []
    for node in getattr(response, "source_nodes", []):
        metadata = node.metadata or {}
        page_value = metadata.get("page")
        if isinstance(page_value, str) and page_value.isdigit():
            page_value = int(page_value)
        excerpt = _build_excerpt(node.get_content())
        sources.append(
            {
                "source": metadata.get("source"),
                "document_title": metadata.get("document_title"),
                "document_filename": metadata.get("document_filename"),
                "page": page_value,
                "section_title": metadata.get("section_title"),
                "score": getattr(node, "score", None),
                "document_category": metadata.get("document_category"),
                "excerpt": excerpt,
                "chunk_id": metadata.get("chunk_id"),
            }
        )

    return {"answer": cleaned_answer, "sources": sources}


class _AsyncQdrantProxy:
    """Run Qdrant sync calls in a thread to satisfy async interface requirements."""

    def __init__(self, delegate: QdrantClient) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._delegate, name)
        if not callable(attribute):
            return attribute

        async def _async_call(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: attribute(*args, **kwargs))

        _async_call.__name__ = getattr(attribute, "__name__", name)
        return _async_call


def _get_qdrant_client(path: str) -> QdrantClient:
    global _QDRANT_CLIENT, _QDRANT_CLIENT_PATH

    if _QDRANT_CLIENT is not None and _QDRANT_CLIENT_PATH == path:
        return _QDRANT_CLIENT

    if _QDRANT_CLIENT is not None:
        close_fn = getattr(_QDRANT_CLIENT, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass

    _QDRANT_CLIENT = QdrantClient(path=path)
    _QDRANT_CLIENT_PATH = path
    return _QDRANT_CLIENT


def _get_qdrant_async_client(path: str) -> Any:
    global _QDRANT_ASYNC_CLIENT, _QDRANT_ASYNC_CLIENT_PATH

    if _QDRANT_ASYNC_CLIENT is not None and _QDRANT_ASYNC_CLIENT_PATH == path:
        return _QDRANT_ASYNC_CLIENT

    client = _get_qdrant_client(path)

    try:
        async_client: Any = AsyncQdrantClient(path=path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Falling back to threaded async proxy for Qdrant at %s: %s", path, exc)
        async_client = _AsyncQdrantProxy(client)

    _QDRANT_ASYNC_CLIENT = async_client
    _QDRANT_ASYNC_CLIENT_PATH = path
    return _QDRANT_ASYNC_CLIENT
