"""Streamlit front-end for the universal RAG demo."""
from __future__ import annotations

import html
import os
import sys
from base64 import b64encode
from pathlib import Path

import httpx
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_ROOT = Path(__file__).resolve().parent
LOGO_PNG_PATH = UI_ROOT / "assets" / "7RL_logo@8x.png"
LOGO_SVG_PATH = UI_ROOT / "assets" / "7rlines_logo.svg"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    DEFAULT_STOP_SEQUENCES,
    get_prompt_presets,
    get_settings,
    reload_settings,
    save_runtime_settings,
)

# Clear settings cache on startup to get fresh values from file
reload_settings()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

MODEL_SWITCH_OPTIONS = [
    {
        "label": "GPT",
        "candidates": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    },
    {
        "label": "Llama",
        "candidates": ["llama-3.3-70b-versatile"],
    },
    {
        "label": "Kimi",
        "candidates": ["moonshotai/kimi-k2-instruct-0905", "moonshotai/kimi-k2-instruct"],
    },
]

PROMPT_PRESET_LABELS = {
    "custom": "Оставить текущий",
    "universal": "Универсальный (рекомендуется)",
    "concise": "Краткий",
    "strict_citations": "Строгие ссылки на источники",
}


def answer_via_api(question: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{API_BASE_URL}/chat",
            json={"question": question},
        )
        response.raise_for_status()
        return response.json()


def list_documents_via_api() -> dict:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()


def upload_documents_via_api(uploaded_files: list) -> dict:
    files = []
    for item in uploaded_files:
        files.append(
            (
                "files",
                (
                    item.name,
                    item.getvalue(),
                    item.type or "application/octet-stream",
                ),
            )
        )

    with httpx.Client(timeout=600.0) as client:
        response = client.post(f"{API_BASE_URL}/documents", files=files)
        response.raise_for_status()
        return response.json()


def delete_documents_via_api(file_names: list[str]) -> dict:
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{API_BASE_URL}/documents/delete",
            json={"file_names": file_names},
        )
        response.raise_for_status()
        return response.json()


def fetch_groq_models(api_key: str) -> dict[str, str]:
    fallback_models = {
        option["label"]: option["candidates"][0]
        for option in MODEL_SWITCH_OPTIONS
    }
    if not api_key:
        return fallback_models

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{get_settings().groq_api_base}/models", headers=headers)
            response.raise_for_status()
            payload = response.json() or {}
    except httpx.HTTPStatusError as exc:
        st.info(f"Groq API вернул ошибку: {exc.response.status_code}")
        return fallback_models
    except Exception as exc:  # noqa: BLE001
        st.info(f"Не удалось получить список моделей Groq: {exc}")
        return fallback_models

    data = payload.get("data", [])
    active_models = {
        item.get("id")
        for item in data
        if item.get("id") and item.get("active", True)
    }

    selected_models: dict[str, str] = {}
    for option in MODEL_SWITCH_OPTIONS:
        model_id = next((mid for mid in option["candidates"] if mid in active_models), None)
        if model_id:
            selected_models[option["label"]] = model_id

    return selected_models or fallback_models


def get_api_offline_message() -> str:
    return f"API сервер недоступен. Проверьте, что backend запущен: {API_BASE_URL}"


def resolve_model_family(model_id: str) -> str:
    for option in MODEL_SWITCH_OPTIONS:
        if model_id in option["candidates"]:
            return option["label"]
    return "Custom"


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    logo_path: Path | None = None
    if LOGO_PNG_PATH.exists():
        logo_path = LOGO_PNG_PATH
    elif LOGO_SVG_PATH.exists():
        logo_path = LOGO_SVG_PATH

    if logo_path is None:
        return ""
    extension = logo_path.suffix.lower()
    mime_type = "image/svg+xml" if extension == ".svg" else "image/png"
    return f"data:{mime_type};base64,{b64encode(logo_path.read_bytes()).decode('ascii')}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700&display=swap');

            :root {
                --bg-page: #f5f6f9;
                --panel: #ffffff;
                --panel-soft: #fbfbfd;
                --border: #e4e7ee;
                --border-strong: #d7dbe5;
                --text-main: #101828;
                --text-soft: #667085;
                --brand-main: #ff2e2c;
                --brand-strong: #db2928;
            }

            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
                color: var(--text-main);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(900px 380px at -10% -10%, rgba(255, 46, 44, 0.08), rgba(255, 46, 44, 0) 68%),
                    linear-gradient(180deg, #f8f9fc 0%, var(--bg-page) 100%);
            }

            [data-testid="stHeader"] {
                background: rgba(245, 246, 249, 0.88);
                border-bottom: 1px solid #eceff5;
            }

            .main .block-container {
                max-width: 1180px;
                padding-top: 1rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3 {
                font-family: "Sora", "Manrope", sans-serif;
                color: var(--text-main);
            }

            [data-testid="stSidebar"] {
                background: var(--panel);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            .sidebar-brand {
                padding: 0.8rem;
                border: 1px solid var(--border);
                border-radius: 12px;
                background: var(--panel-soft);
                margin-bottom: 0.85rem;
            }

            .sidebar-brand-title {
                font-family: "Sora", "Manrope", sans-serif;
                font-size: 0.95rem;
                color: var(--text-main);
                line-height: 1.2;
                margin: 0;
            }

            .sidebar-brand-subtitle {
                margin: 0.18rem 0 0 0;
                color: var(--text-soft);
                font-size: 0.8rem;
                line-height: 1.2;
            }

            .hero-shell {
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.95rem;
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95)),
                    linear-gradient(130deg, rgba(255, 46, 44, 0.06), rgba(255, 46, 44, 0));
                box-shadow: 0 10px 28px rgba(16, 24, 40, 0.05);
            }

            .hero-top {
                display: grid;
                grid-template-columns: minmax(170px, 220px) minmax(0, 1fr);
                align-items: start;
                column-gap: 1rem;
                min-width: 0;
            }

            .hero-logo {
                width: 100%;
                max-width: 220px;
                height: auto;
                display: block;
                flex-shrink: 0;
            }

            .hero-kicker {
                color: var(--brand-main);
                letter-spacing: 0.13em;
                font-size: 0.68rem;
                font-weight: 700;
                text-transform: uppercase;
            }

            .hero-title {
                font-family: "Sora", "Manrope", sans-serif;
                font-size: clamp(1.25rem, 1.1rem + 0.9vw, 1.85rem);
                font-weight: 700;
                color: var(--text-main);
                margin-top: 0.16rem;
                line-height: 1.15;
            }

            .hero-subtitle {
                margin-top: 0.34rem;
                color: var(--text-soft);
                font-size: 0.9rem;
                line-height: 1.42;
            }

            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.78rem;
            }

            .hero-pill {
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 0.34rem 0.62rem;
                background: var(--panel-soft);
                font-size: 0.8rem;
                color: var(--text-main);
            }

            .hero-pill b {
                color: var(--text-soft);
                font-weight: 600;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.45rem;
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.3rem;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                padding: 0.48rem 0.92rem;
                color: var(--text-soft);
                font-weight: 600;
                font-size: 0.9rem;
            }

            .stTabs [aria-selected="true"] {
                color: #fff !important;
                background: linear-gradient(135deg, var(--brand-main), var(--brand-strong)) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 12px;
                border: 1px solid var(--border);
                background: var(--panel);
            }

            .stButton > button,
            .stFormSubmitButton > button {
                border: 1px solid var(--border-strong);
                border-radius: 10px;
                background: #ffffff;
                color: var(--text-main);
                font-weight: 600;
                transition: all 0.18s ease;
            }

            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                border-color: transparent;
                background: linear-gradient(135deg, var(--brand-main), var(--brand-strong));
                color: #fff;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                border-color: #cfd5e2;
            }

            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {
                filter: brightness(1.03);
                transform: translateY(-1px);
            }

            [data-testid="stTextArea"] textarea,
            [data-testid="stTextInput"] input,
            [data-testid="stNumberInput"] input {
                background: #ffffff !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 10px !important;
                color: var(--text-main) !important;
            }

            [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="tag"] {
                background: #ffffff !important;
                border-color: var(--border-strong) !important;
                color: var(--text-main) !important;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 12px;
                overflow: hidden;
                background: #ffffff;
            }

            [data-testid="stMetricValue"] {
                color: var(--text-main);
            }

            [data-testid="stMetricLabel"] {
                color: var(--text-soft);
            }

            hr {
                border-color: var(--border);
            }

            @media (max-width: 980px) {
                .main .block-container {
                    padding-top: 0.8rem;
                    padding-left: 0.85rem;
                    padding-right: 0.85rem;
                }

                .hero-top {
                    grid-template-columns: minmax(150px, 190px) minmax(0, 1fr);
                    column-gap: 0.8rem;
                }

                .hero-logo {
                    max-width: 190px;
                }
            }

            @media (max-width: 640px) {
                .hero-top {
                    grid-template-columns: 1fr;
                    row-gap: 0.6rem;
                }

                .hero-logo {
                    max-width: 165px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    settings = get_settings()
    logo_data_uri = load_logo_data_uri()
    model_family = resolve_model_family(settings.groq_model)
    rules_state = "ON" if bool(getattr(settings, "domain_rules_enabled", False)) else "OFF"

    docs_total = "n/a"
    try:
        payload = list_documents_via_api()
        docs_total = str(payload.get("total_documents", "n/a"))
    except Exception:
        pass

    logo_html = ""
    if logo_data_uri:
        logo_html = f'<img src="{logo_data_uri}" alt="7R logo" class="hero-logo" />'
    safe_family = html.escape(model_family)
    safe_model = html.escape(settings.groq_model)

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-top">
                {logo_html}
                <div>
                    <div class="hero-kicker">Universal RAG</div>
                    <div class="hero-title">RAG Demo Console</div>
                    <div class="hero-subtitle">
                        Поиск по базе знаний, управление документами и настройка моделей в одном интерфейсе.
                    </div>
                </div>
            </div>
            <div class="hero-meta">
                <span class="hero-pill"><b>Family:</b> {safe_family}</span>
                <span class="hero-pill"><b>Model:</b> {safe_model}</span>
                <span class="hero-pill"><b>Docs:</b> {docs_total}</span>
                <span class="hero-pill"><b>Top-K:</b> {settings.retrieval_top_k}</span>
                <span class="hero-pill"><b>Temp:</b> {settings.temperature:.2f}</span>
                <span class="hero-pill"><b>Rules:</b> {rules_state}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    settings = get_settings()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-brand-title">Universal RAG Demo</p>
            <p class="sidebar-brand-subtitle">Рабочая панель</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.metric("Top-K документов", int(settings.retrieval_top_k))
    st.sidebar.metric("Max tokens", int(settings.max_output_tokens))
    st.sidebar.write(f"**Модель:** `{settings.groq_model}`")
    st.sidebar.write(f"**Температура:** {settings.temperature}")
    st.sidebar.write(f"**Top-p:** {settings.top_p}")
    st.sidebar.write(
        f"**Доменные правила:** {'вкл' if getattr(settings, 'domain_rules_enabled', False) else 'выкл'}"
    )

    try:
        payload = list_documents_via_api()
        st.sidebar.write(f"**Документов в базе:** {payload.get('total_documents', 0)}")
    except Exception:
        st.sidebar.write("**Документов в базе:** недоступно (API offline)")


def render_documents() -> None:
    st.subheader("Документы")
    st.caption("Загрузка или удаление автоматически запускает переиндексацию.")

    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Добавить документы (.pdf, .docx)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )

        upload_disabled = not uploaded_files
        if st.button("Загрузить и переиндексировать", disabled=upload_disabled, type="primary"):
            with st.spinner("Загружаю документы и пересобираю индекс..."):
                try:
                    result = upload_documents_via_api(uploaded_files)
                except httpx.ConnectError:
                    st.error(get_api_offline_message())
                except httpx.HTTPStatusError as exc:
                    st.error(f"Ошибка API: {exc.response.status_code} - {exc.response.text}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось загрузить документы: {exc}")
                else:
                    st.success(
                        f"Загружено: {len(result.get('saved_files', []))}. "
                        f"Всего документов: {result.get('total_documents', 'n/a')}."
                    )
                    st.rerun()

    try:
        payload = list_documents_via_api()
    except httpx.ConnectError:
        st.error(get_api_offline_message())
        return
    except httpx.HTTPStatusError as exc:
        st.error(f"Ошибка API: {exc.response.status_code} - {exc.response.text}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось получить список документов: {exc}")
        return

    documents = payload.get("documents", [])
    st.write(f"**Документов в базе:** {payload.get('total_documents', 0)}")

    if not documents:
        st.info("База документов пока пустая.")
        return

    rows = []
    for item in documents:
        rows.append(
            {
                "file_name": item.get("file_name"),
                "size_kb": round((item.get("size_bytes", 0) or 0) / 1024, 1),
                "modified_at": item.get("modified_at"),
            }
        )
    st.dataframe(rows, width="stretch")

    with st.container(border=True):
        options = [item.get("file_name", "") for item in documents if item.get("file_name")]
        to_delete = st.multiselect("Удалить из базы", options=options)
        delete_disabled = not to_delete
        if st.button("Удалить выбранные и переиндексировать", disabled=delete_disabled):
            with st.spinner("Удаляю документы и пересобираю индекс..."):
                try:
                    result = delete_documents_via_api(to_delete)
                except httpx.ConnectError:
                    st.error(get_api_offline_message())
                except httpx.HTTPStatusError as exc:
                    st.error(f"Ошибка API: {exc.response.status_code} - {exc.response.text}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось удалить документы: {exc}")
                else:
                    if result.get("indexes_cleared"):
                        st.success("Документы удалены. Индекс очищен (база теперь пустая).")
                    else:
                        st.success("Документы удалены и индекс пересобран.")
                    st.rerun()


def render_chat() -> None:
    st.subheader("Чат")
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("chat-form", clear_on_submit=True):
        question = st.text_area("Ваш вопрос", height=120)
        submitted = st.form_submit_button("Спросить")

    if submitted:
        if not question.strip():
            st.warning("Введите вопрос.")
        else:
            with st.spinner("Подготавливаю ответ..."):
                try:
                    result = answer_via_api(question)
                except httpx.ConnectError:
                    st.error(get_api_offline_message())
                except httpx.HTTPStatusError as exc:
                    st.error(f"Ошибка API: {exc.response.status_code} - {exc.response.text}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось получить ответ: {exc}")
                else:
                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": result.get("answer", ""),
                            "sources": result.get("sources", []),
                        }
                    )

    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history), start=1):
            with st.container(border=True):
                st.markdown(f"**Запрос {idx}:** {item['question']}")
                st.markdown(item["answer"])
                sources = item.get("sources", [])
                if sources:
                    with st.expander("Источники"):
                        for source in sources:
                            title = source.get("document_title") or source.get("document_filename") or "Документ"
                            page = source.get("page")
                            section = source.get("section_title")
                            score = source.get("score")
                            parts = [title]
                            if page is not None:
                                parts.append(f"стр. {page}")
                            if section:
                                parts.append(f"раздел: {section}")
                            if score is not None:
                                parts.append(f"score={score:.3f}")
                            st.write(" | ".join(parts))


def render_settings() -> None:
    st.subheader("Настройки модели и промпта")
    settings = get_settings()

    available_models = fetch_groq_models(settings.groq_api_key)
    model_to_label: dict[str, str] = {}
    for label, model_id in available_models.items():
        model_to_label[model_id] = label
    switch_labels = [option["label"] for option in MODEL_SWITCH_OPTIONS if option["label"] in available_models]
    if not switch_labels:
        st.error("Не удалось получить доступные модели GPT/Llama/Kimi.")
        return

    prompt_presets = get_prompt_presets()
    preset_keys = ["custom", *prompt_presets.keys()]
    preset_labels = [PROMPT_PRESET_LABELS.get(key, key) for key in preset_keys]

    with st.form("settings-form"):
        default_label = model_to_label.get(settings.groq_model, switch_labels[0])
        model_label = st.radio(
            "Модель Groq",
            options=switch_labels,
            index=switch_labels.index(default_label),
            horizontal=True,
        )
        model = available_models[model_label]
        st.caption(f"Выбрана модель: `{model}`")
        temperature = st.slider("Температура", 0.0, 1.0, float(settings.temperature), 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, float(settings.top_p), 0.05)
        max_tokens = st.number_input(
            "Максимум токенов",
            min_value=128,
            max_value=4096,
            value=int(settings.max_output_tokens),
            step=50,
        )
        retrieval_top_k = st.slider("Top-K документов", 1, 20, int(settings.retrieval_top_k))

        selected_preset_label = st.selectbox("Пресет промпта", preset_labels, index=0)
        selected_preset_key = preset_keys[preset_labels.index(selected_preset_label)]
        prompt = st.text_area("Системный промпт", value=settings.system_prompt, height=350)

        stop_sequences = settings.stop_sequences or DEFAULT_STOP_SEQUENCES
        stop_text = st.text_input(
            "Стоп-последовательности (через запятую)",
            value=", ".join(stop_sequences),
        )

        reranker_display = [
            ("Без реранкера", "none"),
            ("Keyword overlap (легкий)", "keyword_overlap"),
            ("Cross-encoder русский", "DiTy/cross-encoder-russian-msmarco"),
            ("Cross-encoder MS MARCO (англ.)", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        ]
        current_reranker = settings.reranker_model or "keyword_overlap"
        if settings.reranker_top_n <= 0:
            current_reranker = "none"
        reranker_labels = [label for label, _ in reranker_display]
        reranker_values = {label: value for label, value in reranker_display}
        current_label = next((label for label, value in reranker_display if value == current_reranker), reranker_labels[0])
        reranker_label = st.selectbox("Реранкер", reranker_labels, index=reranker_labels.index(current_label))
        selected_reranker = reranker_values[reranker_label]
        reranker_top_n = settings.reranker_top_n if selected_reranker != "none" else 0
        if selected_reranker != "none":
            max_reranker_top_n = max(1, min(10, int(retrieval_top_k)))
            default_reranker_top_n = max(1, int(reranker_top_n or 4))
            default_reranker_top_n = min(default_reranker_top_n, max_reranker_top_n)
            reranker_top_n = st.slider(
                "Top-N для реранкера",
                min_value=1,
                max_value=max_reranker_top_n,
                value=default_reranker_top_n,
            )

        domain_rules_enabled = st.checkbox(
            "Включить доменные правила (коды операций и спец-бустеры)",
            value=bool(getattr(settings, "domain_rules_enabled", False)),
        )

        submitted = st.form_submit_button("Сохранить настройки")

    if submitted:
        cleaned_stops = [seq.strip() for seq in stop_text.split(",") if seq.strip()]
        final_prompt = prompt.strip()
        if selected_preset_key != "custom":
            final_prompt = prompt_presets[selected_preset_key]

        update = {
            "groq_model": model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_tokens),
            "retrieval_top_k": int(retrieval_top_k),
            "system_prompt": final_prompt,
            "stop_sequences": cleaned_stops or DEFAULT_STOP_SEQUENCES,
            "domain_rules_enabled": bool(domain_rules_enabled),
        }
        if selected_reranker == "none":
            update["reranker_model"] = "none"
            update["reranker_top_n"] = 0
        else:
            update["reranker_model"] = selected_reranker
            update["reranker_top_n"] = int(reranker_top_n)
        save_runtime_settings(update)
        st.success("Настройки сохранены.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Universal RAG Demo", layout="wide")
    inject_styles()
    render_sidebar()
    render_hero()

    tab_chat, tab_documents, tab_settings = st.tabs([
        "Чат",
        "Документы",
        "Настройки",
    ])

    with tab_chat:
        render_chat()

    with tab_documents:
        render_documents()

    with tab_settings:
        render_settings()


if __name__ == "__main__":
    main()
