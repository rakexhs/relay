FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY relay ./relay
COPY evals ./evals
COPY scripts ./scripts
RUN pip install ".[openai,anthropic,opensearch,otlp]"

RUN useradd --create-home relay
USER relay

EXPOSE 8000
CMD ["uvicorn", "relay.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
