# multilingual-RAG
Router agent (decides which specialized agent to use) Retrieval agent (finds relevant documents) Analysis agent (deep reasoning on content) Synthesis agent (combines information) Validation agent (fact-checks and verifies)

```bash
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

