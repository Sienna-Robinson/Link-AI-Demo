from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

RouteMode = Literal["direct_answer", "rag", "tool", "hybrid", "clarify"]
Action = Literal["direct_answer", "rag", "tool","clarify"]

ToolName = Literal["lookup_fault_code"]

class FaultCodeArgs(BaseModel):
    code: str = Field(..., description="OBD_II / Link fault code string, e.g. P0123")

class ToolCall(BaseModel):
    name: ToolName
    args: FaultCodeArgs

class RoutePlan(BaseModel):
    mode: RouteMode
    actions: List[Action] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    rag_query: Optional[str] = None
    rag_collections: List[str] = Field(default_factory=list)

    tool_calls: List[ToolCall] = Field(default_factory=list)

    clarifying_question: Optional[str] = None