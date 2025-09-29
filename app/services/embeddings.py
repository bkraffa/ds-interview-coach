import os
from typing import List, Optional
from openai import OpenAI

_CLIENT = None

def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        # Supports both OpenAI and Azure OpenAI (via base_url + api_version)
        kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        kwargs["api_key"] = api_key
        _CLIENT = OpenAI(**kwargs)
    return _CLIENT


def get_embedding_model() -> dict:
    name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dim = os.getenv("EMBEDDING_DIM", "1536")
    return {"name": name, "dim": dim}

def embed_texts(model_cfg: dict, texts: List[str], batch_size: int = 128) -> List[List[float]]:
    client = _client()
    model = model_cfg["name"]
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out
