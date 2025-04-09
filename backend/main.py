from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # SvelteKit dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Welcome to OpenHands API"}

@app.post("/chat")
async def chat(message: ChatMessage):
    # TODO: Add actual chat logic here
    return {"reply": f"Echo: {message.message}"}
