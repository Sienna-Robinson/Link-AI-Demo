from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .safety.deterministic import deterministic_safety_check
from .router.llm_router import route_with_llm
from .tools.dispatch import run_tools

import time
import uuid

app = FastAPI(title="Link AI Demo", version="0.3")

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
    trace["safety"]["llm_classifier"] = {"skipped": True, "reason": "demo_v0.3"}

    # routing: call LLM-A to output a validated JSON plan. This will be upgraded to an AI agent 
    # state machine in the future as this many if/else statements are ugly (and bad practice lol)
    try:
        plan = route_with_llm(
            message=message,
            conversation_summary=req.conversation_summary,
            user_profile=req.user_profile,
            ecu_context=req.ecu_context
        )
        trace["routing"]["llm_plan"] = plan.model_dump()
    except Exception as e:
        trace["routing"]["llm_plan_error"] = str(e)
        plan = None
    
    if plan is None:
        route = "direct_answer"
        answer = f"(Demo) Router failed, fallback to direct. You said: {message}"
        trace["execution"] = {"performed": "direct_answer_fallback"}
    else:
        route = plan.mode
        trace["routing"]["mode"] = route

        if route == "clarify":
            answer = plan.clarifying_question or "Could you clarify what ECU model and what you’re trying to do?"
            trace["execution"] = {"performed": "clarify"}

        elif route == "tool":
            tool_results = run_tools(plan.tool_calls)
            trace["execution"] = {"performed": "tool", "tool_results": tool_results}

            # no AI agent yet to coordinate output, so just output user-friendly message for now (markdown!! teehee)
            if tool_results["calls"]:
                first = tool_results["calls"][0]["output"]
                if first.get("found"):
                    answer = (
                        f"**{first['code']} - {first.get('title', '')}**\n\n"
                        f"{first.get('summary', '')}\n\n"
                        f"**Common causes:**\n- " + "\n- ".join(first.get("common_causes", [])) + "\n\n"
                        f"**Safe checks:**\n- " + "\n- ".join(first.get("safe_checks", []))
                    )
                else:
                    answer = f"I couldn't find that fault code in the demo database: {first.get('error')}"
            else:
                answer = "No tool calls were executed."

        # user-friendly message
        elif route == "rag":
            trace["execution"] = {"performed": "rag_stub", "rag_query": plan.rag_query}
            answer = f"(Demo) Routed to RAG. Would retrieve docs using query: {plan.rag_query or message}"

        elif route == "hybrid":
            trace["execution"] = {"performed": "hybrid_stub", "rag_query": plan.rag_query, "tool_calls": [tc.model_dump() for tc in plan.tool_calls]}
            answer = "(Demo) Routed to hybrid (tools + RAG). Next: run tools + retrieve docs, then synthesize."

        else:  # direct_answer
            trace["execution"] = {"performed": "direct_answer_stub"}
            answer = f"(Demo) Direct answer mode. You said: {message}"

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