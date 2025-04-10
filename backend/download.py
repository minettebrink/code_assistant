import os
from huggingface_hub import snapshot_download


repo_id = os.getenv("MODEL_REPO_ID", "all-hands/openhands-lm-32b-v0.1")

model_cache_dir = "/model_cache"

snapshot_download(
        repo_id=repo_id,
        local_dir=model_cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,  
    )