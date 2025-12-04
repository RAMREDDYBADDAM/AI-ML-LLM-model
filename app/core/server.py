from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Any, Dict
import os
import traceback

from app.core.chains import answer_financial_question

app = FastAPI(title="Financial RAG API")

# Serve a minimal web UI from the repository `web/` directory.
web_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))
if os.path.isdir(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


class ChatRequest(BaseModel):
    user_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    router: Dict[str, Any]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        result = answer_financial_question(req.question)
        return ChatResponse(
            answer=result.get("answer", ""),
            query_type=result.get("query_type", "UNKNOWN"),
            router=result.get("router", {}),
        )
    except Exception as e:
        # Return a valid ChatResponse with error details
        error_msg = f"Server error: {str(e)}"
        return ChatResponse(
            answer=error_msg,
            query_type="ERROR",
            router={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root_index():
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Financial RAG API is running. No web UI found."}
