FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files first (layer cache)
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies (no dev extras, no venv inside container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Create bin directories
RUN mkdir -p bin/data_outputs bin/models_deployed bin/artifacts bin/docs bin/mlruns bin/scripts bin/secrets

EXPOSE 8080

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
