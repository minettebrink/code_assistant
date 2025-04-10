from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_REPO_ID = "all-hands/openhands-lm-32b-v0.1"
MODEL_DTYPE = torch.bfloat16
USE_ACCELERATE = True

app = FastAPI(title="OpenHands LM API")

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

local_model_dir = "/model_cache"

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        model_config_path = os.path.join(local_model_dir, "config.json")
        if not os.path.exists(model_config_path):
            logger.info(f"Model not found in cache. Downloading from {MODEL_REPO_ID}...")
            try:
                snapshot_download(
                    repo_id=MODEL_REPO_ID,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=1, 
                )
                logger.info(f"Model downloaded to: {local_model_dir}")
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                try:
                    import socket
                    logger.info(f"DNS check - IP for huggingface.co: {socket.gethostbyname('huggingface.co')}")
                except Exception as dns_error:
                    logger.error(f"DNS resolution failed: {dns_error}")
                raise
        else:
            logger.info(f"Using existing model from: {local_model_dir}")

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        logger.info("Tokenizer loaded.")

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

@app.on_event("startup")
async def startup_event():
    load_model()


class ChatMessage(BaseModel):
    message: str
    max_new_tokens: int = 100
    temperature: float = 0.7
   
@app.get("/")
async def root():
    return {"message": f"Welcome to OpenHands API. Model '{MODEL_REPO_ID}' status: {'Loaded' if model else 'Failed to load'}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat")
async def chat(request: ChatMessage):
    if not model or not tokenizer:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not available and could not be loaded.")

    try:
        logger.info(f"Received message: {request.message}")
        inputs = tokenizer(request.message, return_tensors="pt")

        if not USE_ACCELERATE and torch.cuda.is_available():
             inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif not USE_ACCELERATE:
             inputs = {k: v.to("cpu") for k, v in inputs.items()}

        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id 
            )

        reply_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if reply_text.startswith(request.message):
             reply_text = reply_text[len(request.message):].strip()

        logger.info(f"Generated reply: {reply_text}")
        return {"reply": reply_text}

    except Exception as e:
        logger.error(f"Error during chat generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

