import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ecu_fitment.json"

def load_fitment_data() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ecu_fitment.json not found at: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _tokens(s: str) -> set[str]:
    # lower, remove punctuation, split into alnum tokens
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def lookup_ecu_fitment(
        make: str,
        model: str,
        engine_detail: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 5,
) -> Dict[str, Any]:
    """
    Docstring for lookup_ecu_fitment
    
    :param code: Description
    :type code: str
    :return: Description
    :rtype: Dict[str, Any]

    Deterministic lookup for ECU fitment.

    Returns:
      {
        "found": bool,
        "query": {...},
        "matches": [ ... ],
        "error": str | None
      }
    """
    try:
        rows = load_fitment_data()
    except Exception as e:
        return {"found": False, "query": {}, "matches": [], "error": str(e)}
    
    make_l = make.strip().lower()
    model_l = model.strip().lower()
    engine_l = (engine_detail or "").strip().lower()

    def match_row(r: Dict[str, Any]) -> bool:
        if str(r.get("make", "")).strip().lower() != make_l:
            return False
        if str(r.get("model", "")).strip().lower() != model_l:
            return False
        if engine_detail:
            query_toks = _tokens(engine_detail)
            row_toks = _tokens(str(r.get("engine_detail", "")))

            # require some meaningful overlap
            # tune this threshold if needed
            overlap = len(query_toks & row_toks)
            if overlap < 3:
                return False
        if year is not None:
            fy = int(r.get("from_year_id", 0) or 0)
            ty = int(r.get("to_year_id", 9999) or 9999)
            if not (fy <= year <= ty):
                return False
        return True

    matches = [r for r in rows if match_row(r)]

    # sort by year range tightness (more specific first)
    def score(r: Dict[str, Any]) -> int:
        fy = int(r.get("from_year_id", 0) or 0)
        ty = int(r.get("to_year_id", 9999) or 9999)
        span = (ty - fy)
        return span

    matches.sort(key=score)

    if not matches:
        return {
            "found": False,
            "query": {"make": make, "model": model, "engine_detail": engine_detail, "year": year},
            "matches": [],
            "error": "No matching fitment records in the demo dataset."
        }

    # keep results compact
    trimmed = []
    for r in matches[:limit]:
        trimmed.append({
            "sku": r.get("sku"),
            "name": r.get("name"),
            "make": r.get("make"),
            "model": r.get("model"),
            "from_year_id": r.get("from_year_id"),
            "to_year_id": r.get("to_year_id"),
            "engine_detail": r.get("engine_detail"),
            "UDEF": r.get("UDEF"),
            "fitment_notes": r.get("Fitment notes") or r.get("fitment_notes") or "",
            "concat": r.get("concat", ""),
        })

    return {
        "found": True,
        "query": {"make": make, "model": model, "engine_detail": engine_detail, "year": year},
        "matches": trimmed,
        "error": None,
    }