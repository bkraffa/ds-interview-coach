.PHONY: up down ingest eval-retrieval eval-llm dev

up:
	docker compose up -d --build

down:
	docker compose down -v

dev:
	streamlit run app/streamlit_app.py

ingest:
	python scripts/ingest.py --input data/raw --collection $$QDRANT_COLLECTION

eval-retrieval:
	python scripts/run_retrieval_eval.py

eval-llm:
	python scripts/run_llm_eval.py
