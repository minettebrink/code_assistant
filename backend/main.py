from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

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
print(f"Downloading model repository: {MODEL_REPO_ID}...")
# Download the model repository snapshot
local_model_dir = snapshot_download(repo_id=MODEL_REPO_ID)
print(f"Model repository downloaded to: {local_model_dir}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
print("Tokenizer loaded.")

print("Loading model...")
model_load_kwargs = {"torch_dtype": MODEL_DTYPE}
if USE_ACCELERATE:
    # device_map="auto" distributes model layers across available devices (GPUs, CPU)
    # Requires the `accelerate` library: pip install accelerate
    model_load_kwargs["device_map"] = "auto"

try:
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        **model_load_kwargs
    )
    print("Model loaded successfully.")
    if not USE_ACCELERATE:
         # If not using accelerate, manually move to GPU if available
         if torch.cuda.is_available():
             print("Moving model to GPU...")
             model.to("cuda")
             print("Model moved to GPU.")
         else:
             print("No GPU detected, using CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    # Depending on your setup, you might want the app to fail hard here
    # or continue without the model loaded (e.g., return errors on endpoints)
    model = None
    tokenizer = None
    # raise RuntimeError(f"Failed to load model: {e}")


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
    # Basic health check, could be expanded later
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat")
async def chat(request: ChatMessage):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        print(f"Received message: {request.message}")
        # Prepare input for the model
        inputs = tokenizer(request.message, return_tensors="pt")

        # Move inputs to the same device as the model's first parameter if not using accelerate
        if not USE_ACCELERATE and torch.cuda.is_available():
             inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif not USE_ACCELERATE:
             # Ensure inputs are on CPU if model is on CPU
             inputs = {k: v.to("cpu") for k, v in inputs.items()}
        # If using accelerate and device_map="auto", inputs are automatically handled

        print("Generating response...")
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


        print(f"Generated reply: {reply_text}")
        return {"reply": reply_text}

    except Exception as e:
        print(f"Error during chat generation: {str(e)}")
        # Consider logging the full traceback here
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

