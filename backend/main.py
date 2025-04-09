from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_REPO_ID = "all-hands/openhands-lm-32b-v0.1"
# Use bfloat16 for reduced memory usage if your hardware supports it, otherwise use float16 or float32
MODEL_DTYPE = torch.bfloat16
# Use accelerate for better distribution across GPUs/CPU+GPU
# requires `pip install accelerate`
USE_ACCELERATE = True

# --- FastAPI App Setup ---
app = FastAPI(title="OpenHands LM API")

# Configure CORS
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173" # Default to SvelteKit dev server
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# Paths
local_model_dir = "/model_cache"

# Initialize global variables
model = None
tokenizer = None

# Function to load the model
def load_model():
    global model, tokenizer
    try:
        # Check if model files already exist in cache
        model_config_path = os.path.join(local_model_dir, "config.json")
        if not os.path.exists(model_config_path):
            logger.info(f"Model not found in cache. Downloading from {MODEL_REPO_ID}...")
            try:
                # Try with higher timeout settings and retries
                snapshot_download(
                    repo_id=MODEL_REPO_ID,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=1,  # Try with a single worker to avoid network race conditions
                )
                logger.info(f"Model downloaded to: {local_model_dir}")
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                # Check if Docker's DNS is working
                try:
                    import socket
                    logger.info(f"DNS check - IP for huggingface.co: {socket.gethostbyname('huggingface.co')}")
                except Exception as dns_error:
                    logger.error(f"DNS resolution failed: {dns_error}")
                raise
        else:
            logger.info(f"Using existing model from: {local_model_dir}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        logger.info("Tokenizer loaded.")

        # Load model
        logger.info("Loading model...")
        model_load_kwargs = {"torch_dtype": MODEL_DTYPE}
        if USE_ACCELERATE:
            model_load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            **model_load_kwargs
        )
        logger.info("Model loaded successfully.")
        if not USE_ACCELERATE and torch.cuda.is_available():
            logger.info("Moving model to GPU...")
            model.to("cuda")
            logger.info("Model moved to GPU.")
        elif not USE_ACCELERATE:
            logger.info("No GPU detected, using CPU.")

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# --- API Endpoints ---
class ChatMessage(BaseModel):
    message: str
    # Optional parameters for generation
    max_new_tokens: int = 100
    temperature: float = 0.7
    # Add other generation parameters as needed (top_k, top_p, etc.)

@app.get("/")
async def root():
    return {"message": f"Welcome to OpenHands API. Model '{MODEL_REPO_ID}' status: {'Loaded' if model else 'Failed to load'}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat")
async def chat(request: ChatMessage):
    if not model or not tokenizer:
        # Try loading the model if it's not loaded yet
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not available and could not be loaded.")

    try:
        logger.info(f"Received message: {request.message}")
        # Prepare input for the model
        inputs = tokenizer(request.message, return_tensors="pt")

        # Move inputs to the same device as the model's first parameter if not using accelerate
        if not USE_ACCELERATE and torch.cuda.is_available():
             inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif not USE_ACCELERATE:
             # Ensure inputs are on CPU if model is on CPU
             inputs = {k: v.to("cpu") for k, v in inputs.items()}
        # If using accelerate and device_map="auto", inputs are automatically handled

        logger.info("Generating response...")
        # Generate response
        # Ensure generation happens without tracking gradients
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id # Prevent warning
                # Add other generation parameters here based on ChatMessage
            )

        # Decode the response, skipping special tokens and the input prompt
        # Note: output might contain the input prompt, depending on the model.
        # Adjust slicing [inputs["input_ids"].shape[1]:] if needed.
        reply_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optional: Remove the original prompt from the start of the reply if present
        if reply_text.startswith(request.message):
             reply_text = reply_text[len(request.message):].strip()

        logger.info(f"Generated reply: {reply_text}")
        return {"reply": reply_text}

    except Exception as e:
        logger.error(f"Error during chat generation: {str(e)}")
        # Consider logging the full traceback here
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

