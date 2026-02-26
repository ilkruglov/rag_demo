"""Application configuration management with runtime overrides."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings

DEFAULT_STOP_SEQUENCES = [
    "<think>",
    "</think>",
    "Thinking:",
    "Reasoning:",
    "Analysis:",
    "Рассуждение:",
    "Анализ:",
]

PROMPT_PRESETS: dict[str, str] = {
    "universal": (
        "Ты — универсальный ассистент по документам.\n\n"
        "ЗАДАЧА:\n"
        "- Отвечай только на основе предоставленного контекста.\n"
        "- Если информации недостаточно, прямо скажи: «Я не знаю» и уточни, каких данных не хватает.\n"
        "- Не выдумывай факты и не добавляй сведения вне контекста.\n\n"
        "ФОРМАТ:\n"
        "**Ответ:** Краткий и точный ответ по существу (1-3 абзаца).\n"
        "**Источники:** Перечисли релевантные источники из контекста (название документа, при наличии страница/раздел).\n\n"
        "ОГРАНИЧЕНИЯ:\n"
        "- Не раскрывай внутренние рассуждения.\n"
        "- Не выводи технические поля (chunk_id, source, document_filename и т.п.)."
    ),
    "concise": (
        "Ты — ассистент по документам. Отвечай строго по контексту и максимально кратко.\n\n"
        "Правила:\n"
        "1. Только факты из контекста.\n"
        "2. Если фактов нет — напиши «Я не знаю».\n"
        "3. В конце добавь строку с источниками.\n\n"
        "Формат:\n"
        "**Ответ:** [кратко]\n"
        "**Источники:** [список документов/разделов/страниц]"
    ),
    "strict_citations": (
        "Ты — эксперт по документам. Дай точный ответ только на основании контекста и с обязательными ссылками на источник.\n\n"
        "Правила:\n"
        "1. Любое утверждение должно быть подтверждено найденными фрагментами.\n"
        "2. Если подтверждения нет — напиши «Я не знаю».\n"
        "3. При противоречиях явно укажи оба варианта и их источники.\n\n"
        "Формат:\n"
        "**Ответ:** [структурированный ответ]\n"
        "**Источники:** [документ, страница/раздел]\n"
        "**Ограничения:** [чего не хватает в контексте]"
    ),
}
DEFAULT_SYSTEM_PROMPT = PROMPT_PRESETS["universal"]

RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "runtime_settings.json"
RUNTIME_OVERRIDABLE_KEYS = {
    "system_prompt",
    "groq_model",
    "temperature",
    "top_p",
    "max_output_tokens",
    "stop_sequences",
    "retrieval_top_k",
    "reranker_model",
    "reranker_top_n",
    "force_offline_mode",
    "domain_rules_enabled",
}


class Settings(BaseSettings):
    """Runtime settings loaded from environment and optional overrides."""

    groq_api_key: str = ""  # Required for chat, optional for indexing
    groq_model: str = "qwen/qwen3-32b"
    temperature: float = 0.1
    top_p: float = 0.7
    max_output_tokens: int = 800
    stop_sequences: list[str] = DEFAULT_STOP_SEQUENCES
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    qdrant_path: str = "data/qdrant"
    storage_dir: str = "data/storage"
    qdrant_collection: str = "nrd_documents"
    retrieval_top_k: int = 15
    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "DiTy/cross-encoder-russian-msmarco"
    reranker_top_n: int = 3
    ingest_chunk_size: int = 800
    ingest_chunk_overlap: int = 200
    force_offline_mode: bool = False
    domain_rules_enabled: bool = False
    groq_api_base: str = "https://api.groq.com/openai/v1"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def _load_runtime_overrides() -> dict:
    if not RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError:
        pass
    return {}


@lru_cache
def get_settings() -> Settings:
    base = Settings()
    overrides = _load_runtime_overrides()
    if overrides:
        merged = {**base.model_dump(), **overrides}
        try:
            base = Settings.model_validate(merged)
        except ValidationError:
            base = Settings()  # fall back to defaults if overrides invalid
    return base


def reload_settings() -> None:
    """Clear cached settings so new overrides take effect."""

    get_settings.cache_clear()


def save_runtime_settings(update: dict) -> Settings:
    """Persist selected runtime settings to disk and reload configuration."""

    current = get_settings()
    merged = current.model_dump()
    for key, value in update.items():
        if key in RUNTIME_OVERRIDABLE_KEYS:
            merged[key] = value

    validated = Settings.model_validate(merged)
    data_to_store = {key: getattr(validated, key) for key in RUNTIME_OVERRIDABLE_KEYS}

    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data_to_store, handle, indent=2, ensure_ascii=False)

    reload_settings()

    try:
        from app.services.chat import refresh_query_engine_cache

        refresh_query_engine_cache()
    except Exception:  # noqa: BLE001
        pass

    return get_settings()


def get_raw_documents_dir(settings: Settings | None = None) -> Path:
    """Return the directory used for storing uploaded raw documents."""

    active = settings or get_settings()
    return Path(active.qdrant_path).parent / "raw"


def get_prompt_presets() -> dict[str, str]:
    """Return supported prompt presets for UI/API usage."""

    return dict(PROMPT_PRESETS)
