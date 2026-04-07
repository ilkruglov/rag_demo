"""Application configuration management with runtime overrides and task profiles."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
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

DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_LABEL = "СДЭК"
DEFAULT_PROFILE_DESCRIPTION = "Базовый профиль FAQ СДЭК."

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
    "nrd_depository_support": (
        "Ты — ассистент клиентской поддержки НРД по депозитарной деятельности.\n\n"
        "ЗАДАЧА:\n"
        "- Отвечай только на основе предоставленного контекста из документов НРД.\n"
        "- Помогай оператору поддержки быстро понять порядок действий, требования, ограничения, сроки и необходимые документы.\n"
        "- Если вопрос касается формы, кода операции, основания отказа, участника процесса или последовательности шагов, выделяй это явно.\n"
        "- Если точного ответа в контексте нет, напиши: «Я не знаю» и укажи, какого документа, раздела или данных не хватает.\n"
        "- Не выдумывай нормативные требования и не делай правовых выводов вне контекста.\n\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Сначала дай короткий практический ответ для сотрудника поддержки.\n"
        "2. Затем перечисли, что клиенту или оператору нужно проверить/сделать.\n"
        "3. Если в контексте есть коды операций, формы, приложения или названия разделов — приводи их в ответе.\n"
        "4. При противоречии между документами явно укажи оба варианта и их источники.\n"
        "5. Не раскрывай внутренние рассуждения и технические поля.\n\n"
        "ФОРМАТ:\n"
        "**Ответ:** [кратко и по существу]\n"
        "**Что проверить/сделать:** [список из 1-5 пунктов, если применимо]\n"
        "**Источники:** [документ, раздел, страница/форма/код операции]\n"
        "**Ограничения:** [чего не хватает в контексте, если ответ неполный]"
    ),
}
DEFAULT_SYSTEM_PROMPT = PROMPT_PRESETS["universal"]

RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "runtime_settings.json"
GLOBAL_RUNTIME_OVERRIDABLE_KEYS = {
    "active_profile",
    "profiles",
    "groq_model",
    "temperature",
    "top_p",
    "max_output_tokens",
    "stop_sequences",
    "force_offline_mode",
}
PROFILE_RUNTIME_OVERRIDABLE_KEYS = {
    "system_prompt",
    "qdrant_path",
    "storage_dir",
    "raw_documents_dir",
    "qdrant_collection",
    "retrieval_top_k",
    "reranker_model",
    "reranker_top_n",
    "domain_rules_enabled",
}
PROFILE_METADATA_KEYS = {
    "label",
    "description",
}
RUNTIME_OVERRIDABLE_KEYS = (
    GLOBAL_RUNTIME_OVERRIDABLE_KEYS
    | PROFILE_RUNTIME_OVERRIDABLE_KEYS
    | PROFILE_METADATA_KEYS
)


class ProfileSettings(BaseModel):
    """Overrides for a named task profile."""

    label: str | None = None
    description: str = ""
    system_prompt: str | None = None
    qdrant_path: str | None = None
    storage_dir: str | None = None
    raw_documents_dir: str | None = None
    qdrant_collection: str | None = None
    retrieval_top_k: int | None = None
    reranker_model: str | None = None
    reranker_top_n: int | None = None
    domain_rules_enabled: bool | None = None


class Settings(BaseSettings):
    """Runtime settings loaded from environment and optional overrides."""

    groq_api_key: str = ""  # Required for chat, optional for indexing
    groq_model: str = "qwen/qwen3-32b"
    temperature: float = 0.1
    top_p: float = 0.7
    max_output_tokens: int = 800
    stop_sequences: list[str] = DEFAULT_STOP_SEQUENCES
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    active_profile: str = DEFAULT_PROFILE_ID
    profiles: dict[str, ProfileSettings] = Field(default_factory=dict)
    qdrant_path: str = "data/qdrant"
    storage_dir: str = "data/storage"
    raw_documents_dir: str = ""
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


@lru_cache(maxsize=1)
def _get_env_settings() -> Settings:
    return Settings()


def normalize_profile_id(profile_id: str | None) -> str:
    """Normalize a user-provided profile identifier."""

    value = (profile_id or DEFAULT_PROFILE_ID).strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value).strip("-_")
    return normalized or DEFAULT_PROFILE_ID


def _default_profile_label(profile_id: str) -> str:
    if profile_id == DEFAULT_PROFILE_ID:
        return DEFAULT_PROFILE_LABEL
    return profile_id.replace("-", " ").replace("_", " ").title()


def _default_profile_paths(profile_id: str) -> dict[str, str]:
    if profile_id == DEFAULT_PROFILE_ID:
        return {}

    return {
        "qdrant_path": f"data/profiles/{profile_id}/qdrant",
        "storage_dir": f"data/profiles/{profile_id}/storage",
        "raw_documents_dir": f"data/profiles/{profile_id}/raw",
        "qdrant_collection": f"{profile_id}_documents",
    }


def _normalize_profiles(raw_profiles: dict[str, Any] | None) -> dict[str, ProfileSettings]:
    normalized: dict[str, ProfileSettings] = {}
    for key, value in (raw_profiles or {}).items():
        profile_id = normalize_profile_id(key)
        if profile_id == DEFAULT_PROFILE_ID:
            continue
        try:
            normalized[profile_id] = (
                value
                if isinstance(value, ProfileSettings)
                else ProfileSettings.model_validate(value or {})
            )
        except ValidationError:
            continue
    return normalized


def _load_base_settings() -> Settings:
    base = _get_env_settings()
    overrides = _load_runtime_overrides()
    if overrides:
        merged = {
            **base.model_dump(),
            **{
                key: value
                for key, value in overrides.items()
                if key in GLOBAL_RUNTIME_OVERRIDABLE_KEYS | PROFILE_RUNTIME_OVERRIDABLE_KEYS
            },
        }
        try:
            base = Settings.model_validate(merged)
        except ValidationError:
            base = _get_env_settings()

    normalized_profiles = _normalize_profiles(base.profiles)
    active_profile = normalize_profile_id(base.active_profile)
    if active_profile != DEFAULT_PROFILE_ID and active_profile not in normalized_profiles:
        active_profile = DEFAULT_PROFILE_ID

    return base.model_copy(
        update={
            "active_profile": active_profile,
            "profiles": normalized_profiles,
        }
    )


def profile_exists(profile_id: str | None) -> bool:
    normalized = normalize_profile_id(profile_id)
    if normalized == DEFAULT_PROFILE_ID:
        return True
    return normalized in _load_base_settings().profiles


def get_active_profile_id() -> str:
    return _load_base_settings().active_profile


def get_settings(profile_id: str | None = None) -> Settings:
    base = _load_base_settings()
    resolved_profile_id = normalize_profile_id(profile_id or base.active_profile)

    if resolved_profile_id == DEFAULT_PROFILE_ID:
        return base.model_copy(update={"active_profile": resolved_profile_id})

    if resolved_profile_id not in base.profiles:
        raise KeyError(resolved_profile_id)

    merged = base.model_dump()
    merged.update(_default_profile_paths(resolved_profile_id))
    profile = base.profiles[resolved_profile_id]
    for key, value in profile.model_dump(exclude_none=True).items():
        if key in PROFILE_RUNTIME_OVERRIDABLE_KEYS:
            merged[key] = value

    merged["active_profile"] = resolved_profile_id
    return Settings.model_validate(merged)


def get_profile_catalog() -> dict[str, dict[str, Any]]:
    """Return resolved profile metadata for UI and API."""

    base = _load_base_settings()
    catalog: dict[str, dict[str, Any]] = {}

    default_settings = get_settings(DEFAULT_PROFILE_ID)
    catalog[DEFAULT_PROFILE_ID] = {
        "profile_id": DEFAULT_PROFILE_ID,
        "label": _default_profile_label(DEFAULT_PROFILE_ID),
        "description": DEFAULT_PROFILE_DESCRIPTION,
        "is_active": base.active_profile == DEFAULT_PROFILE_ID,
        "qdrant_path": default_settings.qdrant_path,
        "storage_dir": default_settings.storage_dir,
        "raw_documents_dir": str(get_raw_documents_dir(default_settings)),
        "qdrant_collection": default_settings.qdrant_collection,
        "system_prompt": default_settings.system_prompt,
        "retrieval_top_k": default_settings.retrieval_top_k,
        "reranker_model": default_settings.reranker_model,
        "reranker_top_n": default_settings.reranker_top_n,
        "domain_rules_enabled": default_settings.domain_rules_enabled,
    }

    for profile_id, profile in sorted(base.profiles.items()):
        resolved = get_settings(profile_id)
        catalog[profile_id] = {
            "profile_id": profile_id,
            "label": profile.label or _default_profile_label(profile_id),
            "description": profile.description,
            "is_active": profile_id == base.active_profile,
            "qdrant_path": resolved.qdrant_path,
            "storage_dir": resolved.storage_dir,
            "raw_documents_dir": str(get_raw_documents_dir(resolved)),
            "qdrant_collection": resolved.qdrant_collection,
            "system_prompt": resolved.system_prompt,
            "retrieval_top_k": resolved.retrieval_top_k,
            "reranker_model": resolved.reranker_model,
            "reranker_top_n": resolved.reranker_top_n,
            "domain_rules_enabled": resolved.domain_rules_enabled,
        }

    return catalog


def reload_settings() -> None:
    """Clear cached environment settings so new overrides take effect."""

    _get_env_settings.cache_clear()


def _write_runtime_overrides(data: dict[str, Any]) -> None:
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_runtime_settings(update: dict, profile_id: str | None = None) -> Settings:
    """Persist runtime settings to disk and return resolved settings."""

    raw = _load_runtime_overrides()
    active_profile = normalize_profile_id(update.get("active_profile") or profile_id or get_active_profile_id())

    for key, value in update.items():
        if key in GLOBAL_RUNTIME_OVERRIDABLE_KEYS:
            raw[key] = value

    if profile_id is not None or any(key in PROFILE_RUNTIME_OVERRIDABLE_KEYS | PROFILE_METADATA_KEYS for key in update):
        target_profile = normalize_profile_id(profile_id or get_active_profile_id())
        if target_profile == DEFAULT_PROFILE_ID:
            for key, value in update.items():
                if key in PROFILE_RUNTIME_OVERRIDABLE_KEYS:
                    raw[key] = value
        else:
            current_profiles = {
                key: value.model_dump(exclude_none=True)
                for key, value in _load_base_settings().profiles.items()
            }
            profile_data = dict(current_profiles.get(target_profile, {}))
            for key, value in update.items():
                if key in PROFILE_RUNTIME_OVERRIDABLE_KEYS | PROFILE_METADATA_KEYS:
                    profile_data[key] = value
            profile_data.setdefault("label", _default_profile_label(target_profile))
            current_profiles[target_profile] = profile_data
            raw["profiles"] = current_profiles
            active_profile = target_profile if "active_profile" not in update else normalize_profile_id(update["active_profile"])

    if "active_profile" in update:
        raw["active_profile"] = normalize_profile_id(update["active_profile"])
    elif "profiles" not in raw and active_profile == DEFAULT_PROFILE_ID:
        raw["active_profile"] = DEFAULT_PROFILE_ID
    else:
        raw.setdefault("active_profile", active_profile)

    _write_runtime_overrides(raw)
    reload_settings()

    try:
        from app.services.chat import refresh_query_engine_cache

        refresh_query_engine_cache()
    except Exception:  # noqa: BLE001
        pass

    return get_settings(normalize_profile_id(raw.get("active_profile")))


def save_profile(profile_id: str, update: dict | None = None, activate: bool = False) -> Settings:
    """Create or update a named profile."""

    normalized = normalize_profile_id(profile_id)
    if normalized == DEFAULT_PROFILE_ID:
        result = save_runtime_settings(update or {}, profile_id=DEFAULT_PROFILE_ID)
    else:
        result = save_runtime_settings(update or {}, profile_id=normalized)
    if activate:
        return save_runtime_settings({"active_profile": normalized})

    return result


def delete_profile(profile_id: str) -> str:
    """Delete a named profile and return the resulting active profile id."""

    normalized = normalize_profile_id(profile_id)
    if normalized == DEFAULT_PROFILE_ID:
        raise ValueError("The default profile cannot be deleted")

    if not profile_exists(normalized):
        raise KeyError(normalized)

    raw = _load_runtime_overrides()
    current_profiles = {
        key: value.model_dump(exclude_none=True)
        for key, value in _load_base_settings().profiles.items()
        if key != normalized
    }
    raw["profiles"] = current_profiles

    if normalize_profile_id(raw.get("active_profile")) == normalized:
        raw["active_profile"] = DEFAULT_PROFILE_ID

    _write_runtime_overrides(raw)
    reload_settings()

    try:
        from app.services.chat import refresh_query_engine_cache

        refresh_query_engine_cache()
    except Exception:  # noqa: BLE001
        pass

    return normalize_profile_id(raw.get("active_profile"))


def get_raw_documents_dir(settings: Settings | None = None) -> Path:
    """Return the directory used for storing uploaded raw documents."""

    active = settings or get_settings()
    if getattr(active, "raw_documents_dir", ""):
        return Path(active.raw_documents_dir)
    return Path(active.qdrant_path).parent / "raw"


def get_prompt_presets() -> dict[str, str]:
    """Return supported prompt presets for UI/API usage."""

    return dict(PROMPT_PRESETS)
