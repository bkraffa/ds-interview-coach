"""
Embeddings utilities
"""
import os
from typing import List
from openai import OpenAI

def get_embedding_model():
    """Get embedding model configuration"""
    return {
        "name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "dim": int(os.getenv("EMBEDDING_DIM", "1536"))
    }

def embed_texts(model_cfg: dict, texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """Embed texts using OpenAI API"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model_cfg["name"]
    out: List[List[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    
    return out