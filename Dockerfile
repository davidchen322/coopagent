# Stage 1: resolve dependencies with Poetry -> requirements.txt
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install --no-install-recommends -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml poetry.lock* /build/

# Generate the lock file if missing, then export to a plain requirements file.
RUN if [ ! -f poetry.lock ]; then poetry lock; fi \
    && poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: slim runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

EXPOSE 8000

# Serve the web UI + streaming chat API.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
