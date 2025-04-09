# Single-stage build with model loading at runtime
FROM python:3.11-slim

LABEL stage="application"

WORKDIR /app

# Set environment variables for Hugging Face cache
ENV HF_HUB_CACHE=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache
ENV PYTHONUNBUFFERED=1

# Create the non-root user first
RUN useradd -m -u 1001 appuser

# Install tools needed for troubleshooting
RUN apt-get update && apt-get install -y dnsutils curl iputils-ping net-tools

# Copy requirements file first for caching
COPY requirments.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirments.txt

# Create directories and set initial ownership
RUN mkdir -p /model_cache && \
    chown -R appuser:appuser /app /model_cache

# Copy application code
COPY . /app/backend/

# Ensure final ownership is correct for app code and copied files
RUN chown -R appuser:appuser /app /model_cache

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Command to run the application with model downloading at runtime
CMD ["python", "-c", "import os; os.makedirs('/model_cache', exist_ok=True); print('Starting server...'); import subprocess; subprocess.run(['uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'])"]