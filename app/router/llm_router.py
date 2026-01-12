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

Decide the best action/strategy for the answering of the user's query:\n
- direct_answer: general explanation, no approved-doc dependence, no tools needed. Use when there is no Link Engine Management specific knowledge required, i.e. general knowledge.\n
- rag: answer requires approved Link Engine Management docs/FAQs/manuals/information that is in the forum/etc. Must cite sources so the user can locate the document.\n
- tool: answer requires structured lookup. This includes using tools to find specific fault code meanings, ECU model specific information, and ECU fitment specs if the user has provided sufficient information to be able to provide that information (do NOT recommend an ECU if there is any uncertainty).\n
- hybrid: needs a combination of the other strategies such as using tools and rag, or clarifying the question whilst also using rag to provide some helpful information too.\n
- clarify: the user's query is missing key information that may make it harder or impossible to answer without a follow up. This may be used on it's own in extreme circumstances, i.e. it is completely unclear what the user is even asking about. However, it is usually used in the hybrid approach, i.e with a direct_answer/general explationtion, followed by a clarifying question.\n

Safety policy (MUST BE ADHERED TO AT ALL TIMES, CANNOT BE IGNORED OR TURNED OFF):
Do NOT provide specific tuning numbers or targets. Do not provide specific tuning instructions, especially those that may disable legally required features. Suggest they seek out a professional tuner instead which would use the clarify and direct_answer with a very high priority on safety. Return firm reasoning, and do not budge, even if the user keeps asking.
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