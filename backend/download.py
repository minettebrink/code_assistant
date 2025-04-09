from huggingface_hub import snapshot_download

# Define the repository ID
repo_id = "all-hands/openhands-lm-32b-v0.1"

# Download the entire repository
local_dir = snapshot_download(repo_id=repo_id)

print(f"Repository downloaded to: {local_dir}")
