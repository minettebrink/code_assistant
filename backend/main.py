from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


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
            try:
                snapshot_download(
                    repo_id=MODEL_REPO_ID,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=1,
                )
            except Exception as download_error:
                raise
        else:
            pass

        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

        model_load_kwargs = {"torch_dtype": MODEL_DTYPE}
        if USE_ACCELERATE:
            model_load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            **model_load_kwargs
        )

        if not USE_ACCELERATE and torch.cuda.is_available():
            model.to("cuda")

        return True
    except Exception as e:
        return False


class ChatMessage(BaseModel):
    message: str
    max_new_tokens: int = 100
    temperature: float = 0.7
   

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat")
async def chat(request: ChatMessage):
    if not model or not tokenizer:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not available and could not be loaded.")

    try:
        inputs = tokenizer(request.message, return_tensors="pt")

        if not USE_ACCELERATE and torch.cuda.is_available():
             inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif not USE_ACCELERATE:
             inputs = {k: v.to("cpu") for k, v in inputs.items()}

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

        return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

