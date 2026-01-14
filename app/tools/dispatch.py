from typing import Any, Dict, List

from app.router.schemas import ToolCall
from .fault_codes import lookup_fault_code
from .ecu_fitment import lookup_ecu_fitment

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

        elif call.name == "lookup_ecu_fitment":
            if not hasattr(call.args, "make"):
                results["errors"].append(
                    f"Bad args for lookup_ecu_fitment: {call.model_dump()}"
                )
                continue
            out = lookup_ecu_fitment(
                make=call.args.make, 
                model=call.args.model,
                engine_detail=call.args.engine_detail,
                year=call.args.year,
            )
            results["calls"].append({
                "name": call.name,
                "args": {
                    "make": call.args.make,
                    "model": call.args.model, 
                    "engine_detail": call.args.engine_detail,
                    "year": call.args.year,
                },
                "output": out,
            })
        else:
            results["errors"].append(f"Tool not allowed: {call.name}")

    return results