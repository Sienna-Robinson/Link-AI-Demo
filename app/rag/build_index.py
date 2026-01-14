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
OUT_PATH = Path(__file__).resolve().parent / "index.jsonl"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# simple chunking
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

def read_text_files() -> List[Dict[str, Any]]:
    docs = []
    for pattern in DOCS_DIR.rglob("*"):
        if pattern.is_file() and pattern.suffix.lower() in [".md", ".txt"]:
            text = pattern.read_text(encoding="utf-8", errors="ignore")
            docs.append({"doc_id": pattern.name, "path": str(pattern), "text": text})
    return docs

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[Dict[str, Any]]:
    chunks = []
    n = len(text)
    start = 0
    index = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id": index,
                "start_char": start,
                "end_char": end,
                "text": chunk
            })
            index += 1
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    # openai embeddings API: batch inputs
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

def main():
    docs = read_text_files()
    if not docs:
        raise RuntimeError(f"No docs found in {DOCS_DIR}. Add .md or .txt files first.")
    
    rows = []
    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_CHARS, CHUNK_OVERLAP)
        for chunk in chunks:
            rows.append({
                "doc_id": doc["doc_id"],
                "path": doc["path"],
                "chunk_id": chunk["chunk_id"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "text": chunk["text"]
            })
    
    # embed
    embeddings = embed_texts([row["text"] for row in rows])
    for row, embed in zip(rows, embeddings):
        row["embedding"] = embed

    # save JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} chunks to {OUT_PATH}")

if __name__ == "__main__":
    main()

