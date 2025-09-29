from typing import List, Dict
import re

def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        # try to break at last period for nicer splits
        last_dot = chunk.rfind(".")
        if last_dot > 400:  # avoid very short tails
            end = start + last_dot + 1
            chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks

def to_payloads(chunks: List[str], source: str, category: str) -> List[Dict]:
    return [{"text": c, "source": source, "category": category} for c in chunks]
