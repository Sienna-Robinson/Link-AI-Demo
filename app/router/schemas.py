from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

RouteMode = Literal["direct_answer", "rag", "tool", "hybrid", "clarify"]

class ToolCall(BaseModel):
    name: str = Field(..., description="Tool identifier")
    args: Dict[str, Any] = Field(default_factory=dict)

class RoutePlan(BaseModel):
    mode: RouteMode
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    rag_query: Optional[str] = None
    rag_collections: List[str] = Field(default_factory=list)

    tool_calls: List[ToolCall] = Field(default_factory=list)

    clarifying_question: Optional[str] = None