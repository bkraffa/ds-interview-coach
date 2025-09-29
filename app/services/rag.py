from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Record
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from .embeddings import get_embedding_model, embed_texts
from typing import List, Dict, Optional

class RagOrchestrator:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection: str):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection = collection
        self.embedder = get_embedding_model()

    def _mode_filter(self, mode: str) -> Optional[Filter]:
        if mode == "all": return None
        return Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=mode)  # "technical" or "behavioral"
                )
            ]
        )

    def retrieve(self, query: str, top_k: int = 6, mode: str = "all") -> List[Dict]:
        v = embed_texts(self.embedder, [query])[0]
        flt = self._mode_filter(mode)
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=v,
            limit=top_k,
            query_filter=flt
        )
        out = []
        for h in hits:
            payload = h.payload or {}
            out.append({
                "id": h.id,
                "score": h.score,
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "category": payload.get("category", "unknown")
            })
        return out
