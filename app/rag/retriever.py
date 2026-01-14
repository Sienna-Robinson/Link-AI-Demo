import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = Path(__file__).resolve().parent / "index.jsonl"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na= 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def load_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise RuntimeError(f"RAG index not found at {INDEX_PATH}. Run build_index.py first.")
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def embed_query(q: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[q]
    )
    return resp.data[0].embedding

def retrieve(q: str, top_k: int = 3) -> Dict[str, Any]:
    rows = load_index()
    q_embed = embed_query(q)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        score = cosine(q_embed, row["embedding"])
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    hits = []
    for score, row in top:
        hits.append({
            "score": score,
            "doc_id": row["doc_id"],
            "path": row["path"],
            "chunk_id": row["chunk_id"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "text": row["text"]
        })
    
    return {
        "query": q,
        "top_k": top_k,
        "hits": hits
    }