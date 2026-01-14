from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field

RouteMode = Literal["direct_answer", "rag", "tool", "hybrid", "clarify"]
Action = Literal["direct_answer", "rag", "tool","clarify"]

class FaultCodeArgs(BaseModel):
    code: str = Field(..., description="OBD_II / Link fault code string, e.g. P0123")

class FitmentArgs(BaseModel):
    make: str
    model: str
    engine_detail: Optional[str] = None
    year: Optional[int] = None

class FaultCodeToolCall(BaseModel):
    name: Literal["lookup_fault_code"]
    args: FaultCodeArgs

class FitmentToolCall(BaseModel):
    name: Literal["lookup_ecu_fitment"]
    args: FitmentArgs

ToolCall = Union[FaultCodeToolCall, FitmentToolCall]

class RoutePlan(BaseModel):
    mode: RouteMode
    actions: List[Action] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    rag_query: Optional[str] = None
    rag_collections: List[str] = Field(default_factory=list)

    tool_calls: List[ToolCall] = Field(default_factory=list)

    clarifying_question: Optional[str] = None