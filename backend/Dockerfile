# --- Stage 1: Model Downloader and Splitter ---
FROM python:3.11-slim as downloader

LABEL stage="model-downloader"

WORKDIR /app

# Install necessary libraries for downloading and splitting
RUN pip install --no-cache-dir huggingface_hub>=0.15.1 torch>=2.0.0

# Copy the download script
COPY backend/download_model.py /app/download_model.py

# Define model repo ID (can be overridden at build time)
ARG MODEL_REPO_ID="all-hands/openhands-lm-32b-v0.1"
ENV MODEL_REPO_ID=${MODEL_REPO_ID}

# Run the download script - downloads to /model_cache
# Exit with error if the script fails
RUN python /app/download_model.py || (echo "Model download script failed!" && exit 1)

# --- Split large files ---
# Create directories for chunks and to record original paths
RUN mkdir -p /app/model_chunks /app/original_paths

# Find files larger than 1GB in the cache, record their relative paths, and split them
# Store relative paths (needed for reassembly in the correct location later)
# Split files into chunks in /app/model_chunks/
RUN cd /model_cache && \
    find . -type f -size +1G -printf '%P\n' > /app/original_paths/large_files.txt && \
    while IFS= read -r file_rel_path; do \
      echo "Splitting $file_rel_path ..."; \
      base_name=$(basename "$file_rel_path"); \
      split -b 1900M "$file_rel_path" "/app/model_chunks/${base_name}.part."; \
    done < /app/original_paths/large_files.txt

# Verify chunks exist (optional)
RUN ls -lh /app/model_chunks

# --- Stage 2: Main Application ---
FROM python:3.11-slim

LABEL stage="application"

WORKDIR /app

# Set environment variables for Hugging Face cache
ENV HF_HUB_CACHE=/model_cache
ENV TRANSFORMERS_CACHE=/model_cache
ENV PYTHONUNBUFFERED=1

# Create the non-root user first
RUN useradd -m -u 1001 appuser

# Copy requirements file first for caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories and set initial ownership
RUN mkdir -p /model_cache /app/model_chunks && \
    chown -R appuser:appuser /app /model_cache

# Copy application code
COPY backend/ /app/backend/

# Copy necessary files from the downloader stage
COPY --from=downloader /model_cache /model_cache
COPY --from=downloader /app/original_paths/large_files.txt /app/original_paths/large_files.txt
COPY --from=downloader /app/model_chunks /app/model_chunks

# Remove the original large files from the final image to save space
RUN while IFS= read -r file_rel_path; do \
      echo "Removing original large file: /model_cache/$file_rel_path"; \
      rm -f "/model_cache/$file_rel_path"; \
    done < /app/original_paths/large_files.txt || echo "No large files listed or removal failed."

# Ensure final ownership is correct for app code and copied files
RUN chown -R appuser:appuser /app /model_cache /app/model_chunks /app/original_paths

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Command to run the application:
# 1. Reassemble the large files from chunks into their original locations within /model_cache
# 2. Clean up the chunks and the path list file
# 3. Start uvicorn
CMD ["bash", "-c", "\
    echo '--- Starting Container ---' && \
    if [ -f /app/original_paths/large_files.txt ]; then \
      echo 'Reassembling model files from chunks...' && \
      while IFS= read -r file_rel_path; do \
        base_name=$(basename \"$file_rel_path\"); \
        target_path=\"/model_cache/$file_rel_path\"; \
        echo \"  Reassembling /app/model_chunks/${base_name}.part.* into ${target_path}\"; \
        cat /app/model_chunks/${base_name}.part.* > \"${target_path}\"; \
        if [ $? -ne 0 ]; then echo '  Reassembly failed!'; exit 1; fi; \
      done < /app/original_paths/large_files.txt && \
      echo 'Reassembly complete. Cleaning up chunks...' && \
      rm -rf /app/model_chunks /app/original_paths; \
    else \
      echo 'No large_files.txt found, skipping reassembly.'; \
    fi && \
    echo 'Starting Uvicorn server...' && \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
"]