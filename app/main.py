from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel

import time
import uuid

app = FastAPI(title="Link AI Demo", version="0.1")

class ChatRequest(BaseModel):
    message: str
    conversation_summary: Optional[str]= None
    user_profile: Dict[str, Any] = {}
    ecu_context: Dict[str, Any] = {}
    attachments: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    request_id: str
    route: str
    answer:str
    citations: List[Dict[str, Any]] = []
    telemetry: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # normalise just a little
    message = req.message.strip()

    route = "direct_answer" # temp
    answer = f"(Demo) You said {message}"

    telemetry = {
        "latency_ms": int((time.time() - t0) * 1000),
        "route": route
    }
    return ChatResponse(
        request_id=request_id,
        route=route,
        answer=answer,
        citations=[],
        telemetry=telemetry
    )