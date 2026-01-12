from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .safety.deterministic import deterministic_safety_check

import time
import uuid

app = FastAPI(title="Link AI Demo", version="0.1")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

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
    trace: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # normalise just a little
    raw_message = req.message
    message = (raw_message or "").strip()

    trace: Dict[str, Any] = {
        "input": {
            "message_chars": len(message),
            "has_conversation_summary": bool(req.conversation_summary),
            "user_profile_keys": list(req.user_profile.keys()),
            "ecu_context_keys": list(req.ecu_context.keys()),
            "num_attachments": len(req.attachments)
            },
        "safety": {},
        "routing": {},
        "execution": {}
    }

    # safety (deterministically for now)
    det = deterministic_safety_check(message)
    trace["safety"]["deterministic"] = det
    domain = det["domain"]

    if det["blocked"]:
        route = "refuse_unsafe"
        answer = (
            f"Unfortunately, your request contains wording that is associated with malicious prompt injection (domain: {domain}). " \
            f"I am unable to assist any further with this question. " \
            f"I can help you with a different question, or feel free to contact our support team. " \
        )

        telemetry = {
            "latency_ms": int((time.time() - t0) * 1000),
            "route": route,
            "blocked": True,
        }

        return ChatResponse(
            request_id=request_id,
            route=route,
            answer=answer,
            citations=[],
            telemetry=telemetry,
            trace=trace
        )
    
    # LLM-A safety classifier goes here. hard code skip for now
    trace["safety"]["llm_classifier"] = {"skipped": True, "reason": "demo_v0.1"}

    # routing: call LLM-A to output a validated JSON plan. hard coded direct for now
    route = "direct_answer"
    
    answer = f"(Demo) You said {message}"

    telemetry = {
        "latency_ms": int((time.time() - t0) * 1000),
        "route": route,
        "blocked": False
    }
    return ChatResponse(
        request_id=request_id,
        route=route,
        answer=answer,
        citations=[],
        telemetry=telemetry,
        trace=trace
    )