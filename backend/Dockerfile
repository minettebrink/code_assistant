FROM python:3.11-slim

LABEL stage="application"

WORKDIR /app

ENV HF_HUB_CACHE=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache
ENV PYTHONUNBUFFERED=1


COPY requirments.txt .

RUN pip install --no-cache-dir -r requirments.txt

RUN mkdir -p /model_cache && \
    chown -R appuser:appuser /app /model_cache

COPY . /app/backend/


RUN chown -R appuser:appuser /app /model_cache

USER appuser


EXPOSE 8000


CMD ["python", "-c", "import os; os.makedirs('/model_cache', exist_ok=True); print('Starting server...'); import subprocess; subprocess.run(['uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'])"]