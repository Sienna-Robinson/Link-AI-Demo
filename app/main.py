from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .safety.deterministic import deterministic_safety_check
from .router.llm_router import route_with_llm
from .tools.dispatch import run_tools
from .rag.retriever import retrieve
from .llm.synthesizer import synthesize_with_llm_b

import time
import uuid

CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}

app = FastAPI(title="Link AI Demo", version="0.4")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

class ChatRequest(BaseModel):
    message: str
    session_id: str
    conversation_summary: Optional[str] = None
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

def tool_answer_from_results(tool_results: Dict[str, Any]) -> str:
    if tool_results.get("calls"):
        first = tool_results["calls"][0].get("output", {})
        if first.get("found"):
            return (
                f"**{first['code']} - {first.get('title', '')}**\n\n"
                f"{first.get('summary', '')}\n\n"
                f"**Common causes:**\n- " + "\n- ".join(first.get("common_causes", [])) + "\n\n"
                f"**Safe checks:**\n- " + "\n- ".join(first.get("safe_checks", []))
            )
        return f"I couldn't find that fault code in the demo database: {first.get('error')}"
    return "No tool calls were executed."

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

    session_id = req.session_id

    history = CONVERSATIONS.get(session_id, [])


    citations: List[Dict[str, Any]] = []

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
    trace["safety"]["llm_classifier"] = {"skipped": True, "reason": "demo_v0.4"}

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

        actions = list(plan.actions or [])

        if not actions:
            if route in ("direct_answer", "rag", "tool" "clarify"):
                actions = [route]
            else:
                actions = ["rag"] if plan.rag_query or plan.rag_collections else ["direct_answer"]

        trace["routing"]["actions"] = actions

        tool_results = None
        rag_result = None

        clarify_q = plan.clarifying_question if "clarify" in actions else None

        if "tool" in actions:
            tool_results = run_tools(plan.tool_calls)
            trace["execution"]["tool"] =  tool_results

            for cit in tool_results.get("calls", []):
                citations.append({"type": "tool", "name": cit["name"], "args": cit["args"]})

        if "rag" in actions:
            rag_result = retrieve(plan.rag_query or message, top_k=3)
            trace["execution"]["rag"] = {
                "query": rag_result["query"],
                "top_k": rag_result["top_k"],
                "hits": [
                    {"score": hit["score"], "doc_id": hit["doc_id"], "chunk_id": hit["chunk_id"]} 
                    for hit in rag_result["hits"]
                ],
            }
            # citations for UI
            citations.extend([
                {"type": "rag", "doc_id": hit["doc_id"], "chunk_id": hit["chunk_id"], "score": hit["score"]}
                for hit in rag_result["hits"]
            ])

        answer = synthesize_with_llm_b(
            user_message=message,
            actions=[str(action) for action in actions],
            history=history,
            rag_hits=rag_result["hits"] if rag_result else None,
            tool_results=tool_results,
            clarifying_question=clarify_q,
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        # Keep last N turns (prevent token blowup)
        CONVERSATIONS[session_id] = history[-10:]

        trace["execution"]["performed"] = route

    telemetry = {
        "latency_ms": int((time.time() - t0) * 1000),
        "route": route,
        "blocked": False
    }
    return ChatResponse(
        request_id=request_id,
        route=route,
        answer=answer,
        citations=citations,
        telemetry=telemetry,
        trace=trace
    )