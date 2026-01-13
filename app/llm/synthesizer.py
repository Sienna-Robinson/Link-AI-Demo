import os

from typing import Any, Dict, List, Optional
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

LLM_B_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """You are Link Engine Management's Companion App AI assistant.

Your job:
- Produce a clear, helpful answer for the user.
- Use ONLY the provided information.
- Do NOT hallucinate facts.
- Do NOT provide tuning numbers, targets, or step-by-step tuning instructions.
- If a clarifying question is provided, ask it ONCE at the end.

Style rules:
- Be concise and professional.
- Allow bullet pointing where helpful.
- Include a short disclaimer if information is general.
- Cite sources when RAG content is used.

You are not responsible for routing, tools, or safety decisions. But do try your best to produce the most helpful message with the given facts.
"""

def extract_text(resp) -> str:
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n\n".join(parts).strip()


def synthesize_with_llm_b(
        user_message: str,
        actions: List[str],
        rag_hits: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        clarifying_question: Optional[str] = None
) -> str:
    context_parts: List[str] = []

    if tool_results:
        context_parts.append(
            "TOOL OUTPUTS:\n" +
            str(tool_results)
        )

    if rag_hits:
        rag_text = ""
        for i, hit in enumerate(rag_hits, start=1):
            rag_text += (
                f"[Source {i}] {hit['doc_id']} (chunk {hit['chunk_id']})\n"
                f"{hit['text']}\n\n"
            )
        context_parts.append(
            f"CLARIFYING QUESTION TO ASK:\n{clarifying_question}"
        )
    
    user_prompt = f"""
USER QUESTION:
{user_message}

ACTIONS USED:
{actions}

CONTEXT:
{'\n\n'.join(context_parts)}
"""
    
    resp = client.messages.create(
        model=LLM_B_MODEL,
        max_tokens=700,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    return extract_text(resp)