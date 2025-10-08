# DS Interview Coach — AI-Powered Interview Preparation System

An intelligent RAG-based agent that helps data science candidates prepare for technical and behavioral interviews with personalized coaching and feedback.

## Problem Statement

Data Science interviews are challenging, requiring candidates to master both technical concepts (ML/DL algorithms, statistics, coding) and behavioral skills (STAR responses, communication). This system provides:

- **Personalized Interview Coaching**: Tailored responses based on your specific questions
- **Dual-Mode Preparation**: Separate technical and behavioral interview tracks
- **Smart Retrieval**: Hybrid search combining vector and keyword matching for optimal context
- **Performance Tracking**: Monitor your preparation progress with analytics

## Quick Start

### Prerequisites
- Docker & Docker Compose (v3.9+)
- OpenAI API key
- 4GB+ RAM available
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bkraffa/ds-interview-coach.git
cd ds-interview-coach
```

2. **Set up environment variables**
```bash
cp .env.example infra/.env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start all services**
```bash
cd infra
docker compose up -d --build
```

4. **Wait for services to initialize** (~30 seconds)
```bash
# Check if services are ready
docker compose ps
```

5. **Ingest knowledge base**
```bash
docker compose exec app python scripts/ingest.py --input data/raw
```

6. **Access the applications**
- **Main App**: http://localhost:8501
- **Qdrant UI**: http://localhost:6333/dashboard
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Knowledge Base

The system includes pre-loaded interview questions covering:
- **Machine Learning**: 50+ common ML interview questions
- **Deep Learning**: 111 comprehensive DL questions
- **Behavioral**: Framework templates and sample responses
- **System Design**: Data engineering and MLOps scenarios

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Streamlit UI  │────▶│  RAG Service │────▶│   Qdrant    │
└─────────────────┘     └──────────────┘     └─────────────┘
                              │                      │
                              ▼                      ▼
                        ┌──────────┐          ┌──────────┐
                        │  OpenAI  │          │ Embeddings│
                        └──────────┘          └──────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ Postgres │──────▶ Feedback Storage
                        └──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              ┌──────────┐        ┌──────────┐
              │Prometheus│        │ Grafana  │
              └──────────┘        └──────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `EMBEDDING_MODEL` | OpenAI embedding model | text-embedding-3-small |
| `EMBEDDING_DIM` | Embedding dimensions | 1536 |
| `QDRANT_HOST` | Qdrant host | localhost |
| `QDRANT_PORT` | Qdrant port | 6333 |
| `QDRANT_COLLECTION` | Collection name | interview_chunks |

### Adding Your Own Data

1. Place files in `data/raw/`:
   - PDFs: Technical papers, interview guides
   - CSVs: Question banks, responses
   - TXT/MD: Notes, documentation

2. Run ingestion:
```bash
docker compose exec app python scripts/ingest.py --input data/raw
```

## Monitoring & Analytics

### Grafana Dashboards
Access at http://localhost:3000 (admin/admin)

Available metrics:
- Query response times
- Retrieval accuracy
- User satisfaction scores
- System performance metrics
- Question category distribution

### User Feedback Collection
The system collects feedback through:
- Thumbs up/down ratings
- Detailed feedback forms
- Session analytics

## Evaluation & Testing

### Retrieval Evaluation
```bash
docker compose exec app python scripts/run_retrieval_eval.py
```

Compares:
- Dense retrieval (vector search)
- Sparse retrieval (BM25)
- Hybrid approaches
- Re-ranking strategies

### LLM Evaluation
```bash
docker compose exec app python scripts/run_llm_eval.py
```

Evaluates:
- Response quality
- Prompt variations
- Context window optimization
- Temperature settings

## Development

### Local Development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here

# Run ingestion
python scripts/ingest.py --input data/raw

# Start the app
streamlit run app/streamlit_app.py
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

## Advanced Features

### Hybrid Search
Combines vector similarity with keyword matching:
- Vector search for semantic similarity
- BM25 for exact term matching
- Weighted fusion for optimal results

### Query Rewriting
Automatically enhances user queries:
- Expands abbreviations
- Adds context
- Corrects common misspellings

### Document Re-ranking
Post-retrieval optimization using:
- Cross-encoder models
- Relevance scoring
- Diversity promotion

## Project Structure

```
ds-interview-coach/
├── app/                    # Main application
│   ├── streamlit_app.py   # UI
│   └── services/          # Core services
│       ├── rag.py         # RAG orchestration
│       ├── embeddings.py  # Embedding service
│       ├── chunking.py    # Document chunking
│       ├── reranking.py   # Re-ranking logic
│       └── feedback.py    # Feedback collection
├── scripts/               # Utility scripts
│   ├── ingest.py         # Data ingestion
│   ├── run_retrieval_eval.py
│   └── run_llm_eval.py
├── data/                  # Data directory
│   ├── raw/              # Source documents
│   └── processed/        # Processed data
├── infra/                # Infrastructure
│   ├── docker-compose.yml
│   ├── docker/           # Dockerfiles
│   ├── grafana/          # Dashboards
│   └── prometheus.yml    # Metrics config
├── tests/                # Test suites
├── reports/              # Evaluation reports
└── requirements.txt      # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/newfeature`)
3. Commit changes (`git commit -m 'Add newfeature'`)
4. Push to branch (`git push origin feature/newfeature`)
5. Open a Pull Request (PR)

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Alexey Grigorev and all DataTalks community for all the shared knowledge throughout the last years.

## 📞 Support

- Issues: [GitHub Issues](https://github.com/bkraffa/ds-interview-coach/issues)
- Discussions: [GitHub Discussions](https://github.com/bkraffa/ds-interview-coach/discussions)

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Voice interview simulation
- [ ] Mock interview recordings
- [ ] Integration with LeetCode/HackerRank
- [ ] Custom evaluation metrics
- [ ] Cloud deployment (AWS/GCP/Azure)

---
Built with ❤️ for the Data Science community