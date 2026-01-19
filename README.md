# 🌍 Multilingual Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** system supporting **22+ Indian languages** with **multi-agent architecture** for intelligent document analysis and question answering.

## ✨ Key Features

- 🌐 **Multilingual Support**: English + 22 Indian languages (Hindi, Bengali, Telugu, Tamil, etc.)
- 🤖 **Multi-Agent System**: 6 specialized AI agents (Router, Planner, Retriever, Analyzer, Synthesizer, Validator)
- 📄 **Document Processing**: PDF, DOCX, TXT, CSV with smart chunking
- 🔍 **Hybrid Search**: Vector similarity + BM25 keyword search
- ✅ **Citation Tracking**: Every answer includes source citations
- 🎯 **Query Classification**: Auto-detects query type (QA, Comparison, Summarization, etc.)
- 💰 **Cost Tracking**: Real-time API cost monitoring
- 🚀 **Production Ready**: Docker, monitoring, error handling, logging

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────┐
│           FastAPI REST API                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │  Query   │  │Documents │  │  Health  │      │
│   └──────────┘  └──────────┘  └──────────┘      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│        Multi-Agent Orchestrator (LangGraph)     │
│  ┌──────┐ ┌────────┐ ┌──────────┐ ┌─────────┐   │
│  │Router│→│Planner │→│Retriever │→│Analyzer │   │
│  └──────┘ └────────┘ └──────────┘ └─────────┘   │
│       ┌──────────┐        ┌──────────┐          │
│       │Synthesis │   ←    │Validator │          │
│       └──────────┘        └──────────┘          │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             Core Components                     │
│   ┌─────────────┐  ┌──────────┐  ┌───────────┐  │
│   │ Embeddings  │  │  Qdrant  │  │ Document  │  │
│   │(E5-Large)   │  │(Vector)  │  │ Processor │  │
│   └─────────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────────┘
```
## 📁 Project Structure
```
multilingual-rag/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                      # GitHub Actions CI/CD
│       └── deploy.yml                  # Auto-deployment
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entry point
│   │   ├── config.py                   # Configuration management
│   │   │
│   │   ├── core/                       # Core functionality
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py           # Multilingual embedding system
│   │   │   ├── vector_store.py         # Qdrant integration
│   │   │   ├── document_processor.py   # PDF/DOCX extraction & chunking
│   │   │   ├── language_detector.py    # Language detection
│   │   │   ├── reranker.py             # Document reranking
│   │   │   └── prompts.py              # System prompts for agents
│   │   │
│   │   ├── agents/                     # Multi-agent system (LangGraph)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Base agent class
│   │   │   ├── router.py               # Query routing agent
│   │   │   ├── planner.py              # Query planning agent
│   │   │   ├── retriever.py            # Retrieval agent
│   │   │   ├── analyzer.py             # Analysis agent
│   │   │   ├── synthesizer.py          # Synthesis agent
│   │   │   ├── validator.py            # Validation agent
│   │   │   └── orchestrator.py         # LangGraph orchestration
│   │   │
│   │   ├── api/                        # API layer
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                 # Dependencies (auth, db connections)
│   │   │   ├── middleware.py           # Custom middleware
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── query.py            # Query endpoints
│   │   │       ├── documents.py        # Document management
│   │   │       ├── users.py            # User management
│   │   │       └── health.py           # Health checks
│   │   │
│   │   ├── models/                     # Data models
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py              # Pydantic models (API)
│   │   │   ├── database.py             # SQLAlchemy models
│   │   │   └── enums.py                # Enums (query types, languages)
│   │   │
│   │   ├── services/                   # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── query_service.py        # Query processing logic
│   │   │   ├── document_service.py     # Document management logic
│   │   │   └── user_service.py         # User management logic
│   │   │
│   │   └── utils/                      # Utilities
│   │       ├── __init__.py
│   │       ├── logger.py               # Logging setup
│   │       ├── security.py             # JWT, password hashing
│   │       ├── helpers.py              # Helper functions
│   │       └── exceptions.py           # Custom exceptions
│   │
│   ├── tests/                          # Tests
│   │   ├── __init__.py
│   │   ├── conftest.py                 # Pytest fixtures
│   │   ├── test_embeddings.py
│   │   ├── test_vector_store.py
│   │   ├── test_agents.py
│   │   └── test_api.py
│   │
│   ├── alembic/                        # Database migrations
│   │   ├── versions/
│   │   └── env.py
│   │
│   ├── requirements.txt                # Python dependencies
│   ├── requirements-dev.txt            # Development dependencies
│   └── Dockerfile                      # Docker image (for deployment)
│
├── scripts/                            # Utility scripts
│   ├── setup_cloud_services.sh         # Setup Qdrant, Supabase
│   ├── seed_database.py                # Seed test data
│   ├── run_tests.sh                    # Run all tests
│   └── deploy.sh                       # Deployment script
│
├── data/                               # Sample data (for testing)
│   ├── sample_docs/
│   │   ├── sample_en.pdf
│   │   ├── sample_hi.pdf
│   │   └── sample_mixed.docx
│   └── test_queries.json
│
├── notebooks/                          # Jupyter notebooks (experimentation)
│   ├── 01_test_embeddings.ipynb
│   ├── 02_test_vector_store.ipynb
│   ├── 03_test_agents.ipynb
│   └── 04_demo.ipynb
│
├── docs/                               # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment.md
│   └── development.md
│
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore file
├── .devcontainer/                      # Codespaces configuration
│   └── devcontainer.json
├── docker-compose.yml                  # Docker compose (optional local dev)
├── pyproject.toml                      # Python project config
├── README.md                           # Project documentation
└── LICENSE                             # License file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- OpenRouter API key
- Qdrant Cloud account (free tier)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/multilingual-rag.git
cd multilingual-rag
```

### 2. Setup Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### 3. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 4. Run Application

**Option A: Direct Python**
```bash
cd backend
python -m app.main
```

**Option B: Docker Compose**
```bash
docker-compose up -d
```

### 5. Access API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## 📚 Usage Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "X-User-Id: user123" \
  -F "file=@document.pdf"
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{
    "query": "What is the capital of India?",
    "top_k": 5
  }'
```

### Python Client
```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        headers={"X-User-Id": "user123"},
        files={"file": f}
    )

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    headers={"X-User-Id": "user123"},
    json={"query": "What is AI?", "top_k": 5}
)

result = response.json()
print(result["answer"])
```

## 🧪 Testing
```bash
# Run tests
pytest backend/tests/

# Run with coverage
pytest --cov=app backend/tests/

# Test specific component
python backend/app/core/embeddings.py
python backend/app/agents/orchestrator.py
```

## 📊 Performance

- **Query Processing**: 2-4 seconds average
- **Cost per Query**: $0.005 - $0.01 (optimized with Gemini Flash + Claude Sonnet)
- **Supported Documents**: PDF, DOCX, TXT, CSV (up to 50MB)
- **Concurrent Requests**: 100+ (with proper scaling)

## 🛠️ Tech Stack

**Backend**
- FastAPI 0.109
- LangGraph (Multi-agent orchestration)
- LangChain (RAG pipeline)

**AI/ML**
- OpenRouter (LLM Gateway)
- Claude Sonnet 3.5 (Analysis & Synthesis)
- Gemini Flash 2.0 (Routing & Validation)
- multilingual-e5-large (Embeddings)

**Databases**
- Qdrant (Vector database)
- PostgreSQL (Metadata)
- Redis (Caching)

**DevOps**
- Docker & Docker Compose
- Nginx (Reverse proxy)
- Prometheus + Grafana (Monitoring)



## 🌟 Advanced Features

### Custom Model Selection

Change models per agent in `.env`:
```bash
MODEL_ROUTER=google/gemini-2.0-flash-exp:free
MODEL_ANALYZER=anthropic/claude-3.5-sonnet
MODEL_SYNTHESIZER=openai/gpt-4-turbo
```

### Metadata Filtering
```python
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "Find sales data",
        "filters": {
            "category": "finance",
            "year": "2024"
        }
    }
)
```


## 📄 License

MIT License - see [LICENSE](LICENSE) file


⭐ **Star this repo if you find it helpful!**

--------

🔗 Author

👨‍💻 **Narendran Karthikeyan**

📎 [LinkedIn](https://github.com/iamnarendran) | [GitHub](https://www.linkedin.com/in/narendran-karthikeyan%F0%9F%8C%B3-95862423b)|

------

