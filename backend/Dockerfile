# Stage 1: Download model
FROM python:3.11-slim AS model-downloader

ENV HF_HUB_CACHE=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache

RUN pip install --no-cache-dir huggingface_hub

# Create model cache directory
RUN mkdir -p /model_cache

# Copy just the download script
COPY download.py /app/download.py

# Run the model download script
RUN cd /app && python download.py

# Stage 2: Application image
FROM python:3.11-slim

LABEL stage="application"

WORKDIR /app

ENV HF_HUB_CACHE=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache
ENV PYTHONUNBUFFERED=1

# Create the non-root user
RUN useradd -m -u 1001 appuser

# Copy requirements and install dependencies
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt

# Create model cache directory
RUN mkdir -p /model_cache

# Copy application code
COPY . /app/backend/

# Copy the model from the first stage
COPY --from=model-downloader /model_cache /model_cache

# Set permissions
RUN chown -R appuser:appuser /app /model_cache

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["python", "-c", "import os; os.makedirs('/model_cache', exist_ok=True); print('Starting server...'); import subprocess; subprocess.run(['uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'])"]