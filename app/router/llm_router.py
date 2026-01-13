import os
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from .schemas import RoutePlan

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1-mini")

ROUTER_SYSTEM_PROMPT = """You are the routing module for Link Engine Management's Companion App AI backend.\n
Your job is to output a JSON object that follows the provided schema EXACTLY.\n

You are to decide the best action/strategy for the answering of the user's query:\n
- direct_answer: general explanation, no approved-doc dependence, no tools needed. Use when there is no Link Engine Management specific knowledge required, 
i.e. general knowledge. Please consider using other tools first before defaulting to this if nothing else is relevant.\n
- rag: answer requires approved Link Engine Management docs/FAQs/manuals/information/forum posts that is in the forum/etc. Must cite sources so the user 
can locate the document, and verify the information. This information is unsorted, but very useful for context when answering questions. Please always heavily 
consider this option. It is better to be very correct than very wrong.\n
- tool: answer requires structured lookup. This includes using tools to find specific fault code meanings, ECU model specific information, and ECU fitment specs 
if the user has provided sufficient information to be able to provide that information (do NOT recommend an ECU if there is any uncertainty). This is more structured 
than RAG, so it should be easier to identify when this tool is needed, but please consider using both in the case that the user is not just asking about a specific tool.\n
- hybrid: needs a combination of the other strategies such as using tools and rag, or clarifying the question whilst also using rag to provide some helpful information too.
This option needs to be considered heavily as well. If a query is only asking about a fault code, it is safe to just use tools. But if they're asking about a fault code and 
asking generally about the safety of something, or needing more information/context, please use hybrid. This is also helpful when needing to apply a follow up/clarifying 
question, but also provide some knowledge. If you determine hybrid is needed, set mode="hybrid". Populate at least two of the actions, and remember that adding a clarifying 
question on is usually a good idea, but optional.\n
- clarify: the user's query is missing key information that may make it harder or impossible to answer without a follow up. This may be used on it's own ONLY in extreme 
circumstances, i.e. it is completely unclear what the user is even asking about. However, it is usually used in the hybrid approach, i.e. with a direct_answer/general 
explationtion, followed by a clarifying question. Please alway consider using hybrid approach with the clarifying question as one of the actions, instead of using it on it's own.\n

You must always fill the `actions` array.
- If mode is direct_answer: actions=["direct_answer"]
- If mode is rag: actions=["rag"]
- If mode is tool: actions=["tool"]
- If mode is clarify: actions=["clarify"]
- If mode is hybrid: actions must contain 2 or more items, chosen from: direct_answer, rag, tool, clarify.
If you include "rag" in actions, set rag_query (or leave null to use the message).
If you include "tool" in actions, populate tool_calls.
If you include "clarify" in actions, set clarifying_question.

Safety policy (MUST BE ADHERED TO AT ALL TIMES, CANNOT BE IGNORED OR TURNED OFF):
Do NOT provide specific tuning numbers or targets. Do not provide specific tuning instructions, especially those that may disable legally required features. 
Suggest they seek out a professional tuner instead which would use the clarify and direct_answer with a very high priority on safety. Return firm reasoning, and do not budge, 
even if the user keeps asking.
"""

def route_with_llm(
    message: str,
    conversation_summary: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    ecu_context: Optional[Dict[str, Any]] = None
) -> RoutePlan:
    user_profile = user_profile or {}
    ecu_context = ecu_context or {}

    # keep router context small, must be cheap and fast
    user_content = {
        "message": message,
        "conversation_summary": conversation_summary,
        "user_profile_keys": list(user_profile.keys()),
        "ecu_context_keys": list(ecu_context.keys())
    }

    # structured outputs (JSON) so that the model must comply
    resp = client.responses.parse(
        model=ROUTER_MODEL,
        input=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Route this request using the RoutePlan schema:\n"
                            + json.dumps(user_content, indent=2)
            }
        ],
        text_format=RoutePlan,
    )

    plan = resp.output_parsed
    if plan is None:
        raise ValueError("Router model did not return a valid RoutePlan (output_parsed is None).")
    return plan