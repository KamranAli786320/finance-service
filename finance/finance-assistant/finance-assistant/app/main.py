import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

from app.routers import chat, banking

app = FastAPI(
    title="AI Personal Finance Assistant",
    description="LangGraph + RAG powered personal finance chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(banking.router, prefix="/api")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")}

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))