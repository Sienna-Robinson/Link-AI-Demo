import os
import json
import math
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

DOCS_DIR = Path(__file__).resolve().parent / "docs"
OUT_DIR = Path(__file__).resolve().parent / "index.jsonl"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# simple chunking
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

def read_text_files() -> List[Dict[str, Any]]:
    docs = []
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".md", ".txt"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"doc_id": p.name, "path": str(p), "text": text})
    return docs

def chunk_text
