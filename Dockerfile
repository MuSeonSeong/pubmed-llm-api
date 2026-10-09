FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    MODELS_DIR=/app/models \
    PATH=/app/.venv/bin:$PATH
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY models.json ./models.json
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
