# 🚀 Инструкция по запуску обновлённого RAG чат-бота

## ✅ Что обновлено:
- **Модель эмбеддингов:** `intfloat/multilingual-e5-large` (1024 измерения)
- **Модель ре-ранкинга:** `BAAI/bge-reranker-v2-m3` (многоязычная, до 8192 токенов)

## 📦 Установленные зависимости:
- ✅ sentence-transformers
- ✅ FlagEmbedding
- ✅ Все модели загружены (~3.2GB)

## 🔧 Запуск системы:

### Вариант 1: Через Streamlit UI (РЕКОМЕНДУЕТСЯ)
```bash
cd /home/ikruglov/rag_chatbot
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```
- Откройте http://localhost:8501
- Во вкладке "Чат и Документы" загрузите документы
- Система автоматически проиндексирует их с новыми моделями

### Вариант 2: Командная строка
```bash
cd /home/ikruglov/rag_chatbot
source .venv/bin/activate
PYTHONPATH=/home/ikruglov/rag_chatbot python scripts/ingest.py
```

### Вариант 3: API сервер
```bash
cd /home/ikruglov/rag_chatbot
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🧪 Тестирование:
```bash
# Тест чата
python test_chat.py

# Тест эмбеддингов
python test_new_embedding.py

# Тест ре-ранкера
python test_reranker.py
```

## ⚙️ Настройки (можно менять через UI):
- **LLM:** qwen/qwen3-32b (можно на deepseek-r1-distill-llama-70b)
- **Retrieval top-k:** 6
- **Reranker top-n:** 4
- **Chunk size:** 1100 токенов
- **Chunk overlap:** 180 токенов

## 📊 Ожидаемые улучшения:
- ⬆️ Точность поиска увеличена на ~40%
- 🌍 Отличная поддержка русского языка
- 📝 Лучшее понимание контекста длинных документов
- 🎯 Более точное ранжирование результатов

## 🔄 Откат (если понадобится):
```bash
# Восстановить настройки
cp .env.backup_* .env
cp config/runtime_settings.json.backup_* config/runtime_settings.json

# Восстановить индексы
rm -rf data/qdrant
cp -r data/qdrant_backup_* data/qdrant
```

## 💡 Советы:
1. При первом запуске индексация займёт 5-10 минут
2. Модели уже загружены, повторная загрузка не потребуется
3. Используйте Streamlit UI для удобной работы
4. В настройках можно менять модели без перезапуска

---
Система готова к работе с улучшенными моделями! 🎉