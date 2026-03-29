FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency specification
COPY pyproject.toml .
COPY README.md .

# Install all production dependencies (no dev extras)
# --system writes into the base Python environment so uvicorn is on PATH
RUN uv pip install --system -e "."

# Copy application source
COPY src/ src/

# Expose FastAPI port
EXPOSE 8000

# ChromaDB and data are mounted at runtime via docker-compose volumes,
# so they are not baked into the image.
CMD ["uvicorn", "quaestor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
