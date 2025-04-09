import os
import logging
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the repository ID
repo_id = os.getenv("MODEL_REPO_ID", "all-hands/openhands-lm-32b-v0.1")

# Model directory
model_cache_dir = "/model_cache"
os.makedirs(model_cache_dir, exist_ok=True)

logger.info(f"Starting download of model: {repo_id}")
try:
    # Use parameters to improve resilience
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=model_cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,  # Lower for more stability
    )
    logger.info(f"Model downloaded successfully to: {local_dir}")
except Exception as e:
    logger.error(f"Download failed: {e}")
    # Print network diagnostic info
    try:
        import socket
        logger.info(f"DNS check - IP for huggingface.co: {socket.gethostbyname('huggingface.co')}")
    except:
        logger.error("DNS resolution check failed")
    
    raise
