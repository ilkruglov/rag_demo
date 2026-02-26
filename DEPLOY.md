# Развертывание RAG Chatbot (GPU + Claude Code)

## Требования к системе

- **ОС:** Ubuntu 22.04+ / Debian 12+ / WSL2
- **Python:** 3.11+ (рекомендуется 3.12)
- **GPU:** NVIDIA с CUDA 11.8+ (минимум 4 GB VRAM)
- **RAM:** 8+ GB
- **Диск:** ~5 GB для зависимостей + модели
- **Claude Code:** установлен и настроен

## Быстрый старт с Claude Code

После копирования архива на целевую машину, откройте терминал и запустите Claude Code:

```bash
claude
```

Затем попросите:
```
Распакуй архив ~/rag_chatbot_deploy.tar.gz, установи зависимости с GPU поддержкой и запусти приложение
```

Claude Code выполнит все шаги автоматически.

---

## Ручная установка

### Шаг 1: Проверка GPU

```bash
nvidia-smi
```

Должна отобразиться информация о GPU и версии CUDA.

### Шаг 2: Распаковка

```bash
cd ~
tar -xzvf rag_chatbot_deploy.tar.gz
cd rag_chatbot
```

### Шаг 3: Создание окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Шаг 4: Установка PyTorch с CUDA

```bash
# Для CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Шаг 5: Установка остальных зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 6: Настройка .env

```bash
cat > .env << 'EOF'
GROQ_API_KEY=ваш_ключ_groq
EOF
```

Получить ключ: https://console.groq.com

### Шаг 7: Проверка индекса

```bash
python -c "
from app.config import get_settings
from qdrant_client import QdrantClient
settings = get_settings()
client = QdrantClient(path=settings.qdrant_path)
info = client.get_collection(settings.qdrant_collection)
print(f'Векторов в индексе: {info.points_count}')
"
```

Ожидаемый результат: `Векторов в индексе: 1393`

### Шаг 8: Запуск

```bash
# Streamlit UI (рекомендуется):
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Или FastAPI:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Конфигурация моделей

Текущая конфигурация (app/config.py):

| Компонент | Модель | Описание |
|-----------|--------|----------|
| Embeddings | `intfloat/multilingual-e5-large` | 1024 dim, оптимизирована для русского |
| Reranker | `DiTy/cross-encoder-russian-msmarco` | Русскоязычный cross-encoder |
| LLM | `qwen/qwen3-32b` (Groq) | Генерация ответов |

## Структура проекта

```
rag_chatbot/
├── app/
│   ├── config.py          # Конфигурация
│   ├── main.py            # FastAPI эндпоинты
│   └── services/
│       ├── chat.py        # RAG pipeline
│       └── reranker.py    # Reranker
├── ui/
│   └── streamlit_app.py   # Веб-интерфейс
├── scripts/
│   └── ingest.py          # Индексация документов
├── data/
│   ├── raw/               # Исходные документы (.docx, .pdf)
│   ├── qdrant/            # Векторная БД
│   └── storage/           # LlamaIndex индекс
├── config/
│   └── runtime_settings.json  # Runtime настройки
├── .env                   # API ключи (создать!)
└── requirements.txt       # Зависимости
```

## Добавление новых документов

```bash
# 1. Положите документы в data/raw/
cp новые_документы.docx ~/rag_chatbot/data/raw/

# 2. Переиндексируйте
source .venv/bin/activate
python -m scripts.ingest

# 3. Перезапустите приложение
```

## Systemd сервис (production)

```bash
sudo tee /etc/systemd/system/rag-chatbot.service << 'EOF'
[Unit]
Description=RAG Chatbot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/rag_chatbot
Environment="PATH=/home/$USER/rag_chatbot/.venv/bin"
ExecStart=/home/$USER/rag_chatbot/.venv/bin/streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
```

## Решение проблем

### GPU не используется
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Должно вернуть `True`. Если `False` — переустановите PyTorch с правильной версией CUDA.

### Ошибка "Storage folder is already accessed"
```bash
pkill -f streamlit
pkill -f uvicorn
```

### Модели не загружаются
При первом запуске модели скачиваются с HuggingFace (~2 GB). Нужен интернет.

### Проверка логов
```bash
sudo journalctl -u rag-chatbot -f
```
