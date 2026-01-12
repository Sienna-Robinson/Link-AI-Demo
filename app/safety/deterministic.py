from typing import Dict, Any

UNSAFE_PROMPT_PATTERNS = [
    "ignore all previous instructions", "disregard your system prompt", "system override", 
    "print the initialisation banner", "pretend you are", "no restrictions", "api key",
    "password", "secret", "admin"
]

def deterministic_safety_check(message: str) -> Dict[str, Any]:
    """
    Docstring for deterministic_safety_check
    
    :param message: Description
    :type message: str
    :return: Description
    :rtype: Dict[str, Any]

    Fast, rule-based safety gate.
    Returns a structured verdict the MCP server can enforce.
    """
    text = message.lower()

    for pattern in UNSAFE_PROMPT_PATTERNS:
        if pattern in text:
            return {
                "blocked": True,
                "risk_level": "high",
                "domain": "prompt_injection",
                "reason": f"matched_pattern:{pattern}"
            }
    
    return {
        "blocked": False,
        "risk_level": "low",
        "domain": "unknown",
        "reason": "no_match!"
    }