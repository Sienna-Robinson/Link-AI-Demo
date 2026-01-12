import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fault_codes.json"

_CODE_RE = re.compile(r"^[A-Z]\d{4}$")

def load_fault_db() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        return {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def lookup_fault_code(code: str) -> Dict[str, Any]:
    """
    Docstring for lookup_fault_code
    
    :param code: Description
    :type code: str
    :return: Description
    :rtype: Dict[str, Any]

    Server-side tool: returns structured info about a fault code.
    Always safe (no tuning specific numbers).
    """
    code = (code or "").strip().upper()

    if not _CODE_RE.match(code):
        return {
            "found": False,
            "code": code,
            "error": "Invalid code format. Expected like P0123."
        }
    
    db = load_fault_db()
    item = db.get(code)

    if not item:
        return {
            "found": False,
            "code": code,
            "error": "Code not found in the current demo database."
        }
    
    return {
        "found": True,
        "code": code,
        "title": item.get("title"),
        "summary": item.get("summary"),
        "common_causes": item.get("common_causes", []),
        "safe_checks": item.get("safe_checks", [])
    }