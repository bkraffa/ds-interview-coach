import argparse
import os
import glob
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.services.embeddings import get_embedding_model, embed_texts
from app.services.chunking import simple_chunk, to_payloads
from PyPDF2 import PdfReader


def read_text(path: Path) -> str:
    """Return text content for PDF/TXT/MD/CSV. For CSV, concatenates string-like columns."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    elif path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() in [".csv"]:
        import pandas as pd
        df = pd.read_csv(path)
        cols = [c for c in df.columns if df[c].dtype == object]
        lines: List[str] = []
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in cols if str(row[c]) != "nan"]
            if parts:
                lines.append(" ".join(parts))
        return "\n".join(lines)
    else:
        # Fallback: try to read as text
        return path.read_text(encoding="utf-8", errors="ignore")


def upsert(qdrant: QdrantClient, collection: str, payloads, vectors, batch_size: int = 512):
    """Upsert to Qdrant in batches to keep memory and payload sizes reasonable."""
    assert len(payloads) == len(vectors)
    points: List[PointStruct] = []
    for i, (p, v) in enumerate(zip(payloads, vectors)):
        points.append(PointStruct(id=None, vector=v, payload=p))  # let Qdrant assign ids
        if len(points) >= batch_size:
            qdrant.upsert(collection_name=collection, points=points)
            points = []
    if points:
        qdrant.upsert(collection_name=collection, points=points)


def ensure_collection(qdrant: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw", help="Folder with PDFs/CSVs/TXT/MD")
    parser.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION", "interview_chunks"))
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    # OpenAI embeddings config (name + dimension) from env
    model_cfg = get_embedding_model()

    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )

    # Find files to ingest
    exts = ("*.pdf", "*.txt", "*.md", "*.csv")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.input, ext)))
    if not files:
        print(f"No files found in {args.input}. Put your sources in that folder.")
        return

    # Ensure collection exists with the expected OpenAI embedding dimension
    ensure_collection(qdrant, args.collection, model_cfg["dim"])

    all_payloads, all_texts = [], []

    for f in files:
        path = Path(f)
        # Heuristic category from filename
        category = "behavioral" if ("Behavioral" in path.name or "HR" in path.name) else "technical"

        text = read_text(path)
        if not text.strip():
            print(f"Skipped (empty): {path.name}")
            continue

        chunks = simple_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)
        payloads = to_payloads(chunks, source=path.name, category=category)

        # Accumulate for a single embed pass per file (fewer API calls)
        all_payloads.extend(payloads)
        all_texts.extend([p["text"] for p in payloads])

        print(f"Ingested {len(chunks)} chunks from {path.name} [{category}]")

    if not all_texts:
        print("Nothing to embed. Exiting.")
        return

    # Embed with OpenAI (batched inside embed_texts)
    vectors = embed_texts(model_cfg, all_texts)

    # Upsert
    upsert(qdrant, args.collection, all_payloads, vectors)
    print(f"Upserted {len(all_payloads)} chunks to collection '{args.collection}'. Done.")


if __name__ == "__main__":
    main()
