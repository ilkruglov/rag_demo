FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r /app/requirements.txt

COPY app /app/app
COPY config /app/config
COPY scripts /app/scripts
COPY ui /app/ui
COPY data /app/data
COPY .env.example /app/.env.example

EXPOSE 8001 8501
