.PHONY: help install run ingest check-db eval-retrieval eval-llm eval-all test

help:
	@echo "Available commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make run              - Run Streamlit app"
	@echo "  make ingest           - Ingest data into Qdrant"
	@echo "  make check-db         - Check PostgreSQL data"
	@echo "  make eval-retrieval   - Run retrieval evaluation"
	@echo "  make eval-llm         - Run LLM evaluation"
	@echo "  make eval-all         - Run all evaluations"
	@echo "  make test             - Run tests"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-up        - Start all services"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - View logs"
	@echo "  make docker-eval      - Run evaluations in Docker"

install:
	pip install -r requirements.txt

run:
	streamlit run app/streamlit_app.py

ingest:
	python scripts/ingest.py

check-db:
	cd infra && docker-compose exec app python scripts/check_postgres.py

eval-retrieval:
	cd infra && docker-compose exec app python scripts/run_retrieval_eval.py

eval-llm:
	cd infra && docker-compose exec app python scripts/llm_run_eval.py

eval-all: eval-retrieval eval-llm
	@echo "✅ All evaluations complete! Check reports/ folder"

test:
	cd infra && docker-compose exec app pytest tests/

# Docker commands
docker-up:
	cd infra && docker-compose up -d

docker-down:
	cd infra && docker-compose down

docker-logs:
	cd infra && docker-compose logs -f app

docker-build:
	cd infra && docker-compose build --no-cache

docker-restart:
	cd infra && docker-compose restart app

docker-rebuild:
	cd infra && docker-compose down
	cd infra && docker-compose build --no-cache
	cd infra && docker-compose up -d

docker-eval: eval-retrieval eval-llm
	@echo "All evaluations complete!"

docker-check-db: check-db

docker-psql:
	cd infra && docker-compose exec postgres psql -U postgres -d interview_coach

docker-shell:
	cd infra && docker-compose exec app bash