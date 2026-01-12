from typing import Any, Dict, List

from app.router.schemas import ToolCall
from .fault_codes import lookup_fault_code

def run_tools(tool_calls: List[ToolCall]) -> Dict[str, Any]:
    """
    Executes approved tools only.
    Returns a dict of tool results.
    """
    results: Dict[str, Any] = {"calls": [], "errors": []}

    for call in tool_calls:
        if call.name == "lookup_fault_code":
            out = lookup_fault_code(call.args.code)
            results["calls"].append({
                "name": call.name,
                "args": {"code": call.args.code},
                "output": out
            })
        else:
            results["errors"].append(f"Tool not allowed: {call.name}")

    return results