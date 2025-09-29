# DS Interview Coach — Starter

A minimal, container-ready scaffold for a RAG agent that helps candidates prepare for data‑science interviews (technical + behavioral).

## Quickstart (local)

1. Set openAI key on .env
   - `OPENAI_API_KEY=...`

2. Install deps and run:
   ```bash
   pip install -r requirements.txt
   make ingest

## With Docker Compose

```bash
cd infra
docker compose up -d --build
```

Visit:
- App: http://localhost:8501
- Qdrant: http://localhost:6333
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Ingest your sources
Put PDFs/CSVs into `data/raw/` and run:

```bash
make ingest
```

Then ask questions in the app UI.

## Structure
- `app/` — Streamlit UI + services (`rag.py`, `embeddings.py`, `chunking.py`)
- `scripts/ingest.py` — chunk, embed, upsert to Qdrant
- `infra/` — Docker, Prometheus, Grafana, Compose
- `data/` — raw sources (seeded with your uploaded files)
- `tests/` — smoke tests

Next steps: add BM25 hybrid, reranking, query rewrite, Airflow DAGs, metrics, and evaluation scripts.
