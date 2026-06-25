# Canonical runtime is pinned to Python 3.12 (the host may run a newer interpreter
# that lacks wheels for some optional ML extras). The default install is light:
# the hashing embedder needs no torch, so the image stays small and free-tier safe.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STYLIST_LLM=fake \
    STYLIST_EMBEDDER=hashing

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY core ./core
COPY app ./app
COPY eval ./eval
RUN pip install --upgrade pip && pip install ".[anthropic]"

# Bundle the committed sample fixture (tests/demo run without a live scrape).
COPY data ./data

EXPOSE 8000

# Render/most PaaS inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
