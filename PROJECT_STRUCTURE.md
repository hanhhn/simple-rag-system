# Project Structure - RAG System

## Directory Structure

```
simple-rag-system/
├── src/                                   # Application source code
│   ├── api/                               # API Layer
│   │   ├── __init__.py
│   │   ├── main.py                        # FastAPI application entry point
│   │   ├── dependencies.py               # Dependency injection
│   │   ├── middleware/                   # Custom middleware
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                   # Authentication middleware
│   │   │   ├── logging.py                # Logging middleware
│   │   │   └── rate_limit.py             # Rate limiting middleware
│   │   ├── routes/                       # API route definitions
│   │   │   ├── __init__.py
│   │   │   ├── documents.py              # Document endpoints
│   │   │   ├── query.py                  # Query endpoints
│   │   │   ├── collections.py            # Collection endpoints
│   │   │   ├── tasks.py                  # Task status tracking endpoints
│   │   │   └── health.py                 # Health check endpoints
│   │   └── models/                       # Pydantic models
│   │       ├── __init__.py
│   │       ├── document.py               # Document request/response models
│   │       ├── query.py                  # Query request/response models
│   │       ├── collection.py             # Collection request/response models
│   │       ├── task.py                   # Task status models
│   │       └── common.py                 # Common models
│   │
│   ├── services/                          # Business Logic Layer
│   │   ├── __init__.py
│   │   ├── document_processor.py        # Document processing service
│   │   ├── embedding_service.py         # Embedding generation service
│   │   ├── vector_store.py              # Vector store service (Qdrant)
│   │   ├── llm_service.py               # LLM service (Ollama)
│   │   ├── query_processor.py          # Query processing service
│   │   └── storage_manager.py           # File storage management
│   │
│   ├── tasks/                            # Background Task Processing
│   │   ├── __init__.py
│   │   ├── celery_app.py                # Celery application configuration
│   │   ├── document_tasks.py            # Document processing tasks
│   │   └── embedding_tasks.py           # Embedding generation tasks
│   │
│   ├── core/                             # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py                     # Configuration management
│   │   ├── exceptions.py                 # Custom exceptions
│   │   ├── logging.py                    # Logging setup
│   │   └── security.py                   # Security utilities
│   │
│   ├── utils/                            # Utility functions
│   │   ├── __init__.py
│   │   ├── text_chunker.py             # Text chunking strategies
│   │   ├── validators.py                # Input validators
│   │   ├── text_cleaner.py             # Text cleaning utilities
│   │   └── helpers.py                   # Helper functions
│   │
│   ├── parsers/                         # Document parsers
│   │   ├── __init__.py
│   │   ├── base.py                      # Base parser interface
│   │   ├── pdf_parser.py                # PDF parser
│   │   ├── docx_parser.py               # Word document parser
│   │   ├── txt_parser.py                # Plain text parser
│   │   └── md_parser.py                 # Markdown parser
│   │
│   ├── embedding/                       # Embedding models
│   │   ├── __init__.py
│   │   ├── base.py                      # Base embedding model
│   │   ├── model_loader.py              # Model loader
│   │   ├── cache.py                     # Embedding cache
│   │   └── models/                      # Model implementations
│   │       ├── __init__.py
│   │       ├── minilm.py                # MiniLM model
│   │       └── mpnet.py                 # MPNet model
│   │
│   └── llm/                            # LLM integration
│       ├── __init__.py
│       ├── base.py                      # Base LLM interface
│       ├── ollama_client.py            # Ollama client
│       ├── prompt_builder.py           # Prompt builder
│       ├── templates/                  # Prompt templates
│       │   ├── __init__.py
│       │   ├── rag_template.txt        # RAG prompt template
│       │   └── chat_template.txt       # Chat prompt template
│       └── stream_handler.py           # Stream response handler
│
├── tests/                               # Test suite
│   ├── __init__.py
│   ├── conftest.py                      # Pytest configuration
│   ├── unit/                           # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_document_processor.py
│   │   ├── test_embedding_service.py
│   │   ├── test_vector_store.py
│   │   ├── test_llm_service.py
│   │   ├── test_query_processor.py
│   │   ├── test_parsers.py
│   │   └── test_utils.py
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_document_flow.py
│   │   └── test_query_flow.py
│   └── e2e/                           # End-to-end tests
│       ├── __init__.py
│       ├── test_rag_pipeline.py
│       └── test_user_scenarios.py
│
├── scripts/                            # Utility scripts
│   ├── setup_dev_env.sh               # Development environment setup
│   ├── setup_production_env.sh        # Production environment setup
│   ├── init_qdrant.py                # Initialize Qdrant collections
│   ├── download_models.py            # Download required models
│   ├── start_celery_worker.sh        # Start Celery worker (Linux/Mac)
│   ├── start_celery_worker.bat       # Start Celery worker (Windows)
│   └── cleanup.py                    # Cleanup utility
│
├── deployments/                       # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile                # Application Dockerfile
│   │   ├── docker-compose.yml       # Docker Compose for dev
│   │   └── docker-compose.prod.yml  # Docker Compose for production
│   └── kubernetes/                   # Kubernetes manifests (optional)
│       ├── namespace.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       └── configmap.yaml
│
├── docs/                              # Documentation
│   ├── 01-basic-design.md            # Basic design document
│   ├── 02-c4-model.md               # C4 model diagrams
│   ├── 03-high-level-design.md     # High-level architecture
│   ├── 04-data-flow.md             # Data flow diagrams
│   ├── 05-sequence-diagrams.md     # Sequence diagrams
│   ├── api-documentation.md        # API documentation
│   ├── deployment-guide.md        # Deployment guide
│   └── development-guide.md       # Development guide
│
├── data/                             # Data directories
│   ├── documents/                   # Uploaded documents
│   ├── models/                      # Downloaded models
│   └── cache/                       # Cache directory
│
├── logs/                            # Log files
│   ├── app.log
│   ├── error.log
│   └── access.log
│
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── pyproject.toml                  # Project configuration
├── README.md                       # Project README
└── LICENSE                         # License file
```

## Component Descriptions

### API Layer (`src/api/`)
- **main.py**: FastAPI application entry point, initialization of all components
- **middleware/**: Custom middleware for authentication, logging, and rate limiting
- **routes/**: API endpoint definitions organized by functionality
- **models/**: Pydantic models for request validation and response formatting

### Services Layer (`src/services/`)
- **document_processor.py**: Handles document parsing, text extraction, and chunking
- **embedding_service.py**: Manages embedding generation and model loading
- **vector_store.py**: Interface to Qdrant vector database
- **llm_service.py**: Manages LLM interactions via Ollama
- **query_processor.py**: Orchestrates the RAG pipeline for queries
- **storage_manager.py**: Manages file storage and lifecycle

### Core Layer (`src/core/`)
- **config.py**: Centralized configuration management using Pydantic Settings
- **exceptions.py**: Custom exception classes for error handling
- **logging.py**: Structured logging setup using structlog
- **security.py**: Security utilities (JWT, password hashing, etc.)

### Utilities Layer (`src/utils/`)
- **text_chunker.py**: Various text chunking strategies
- **validators.py**: Input validation utilities
- **text_cleaner.py**: Text cleaning and normalization
- **helpers.py**: Common helper functions

### Parsers Layer (`src/parsers/`)
- **base.py**: Base parser interface
- **pdf_parser.py**: PDF document parser using PyPDF2 or pdfplumber
- **docx_parser.py**: Word document parser using python-docx
- **txt_parser.py**: Plain text parser
- **md_parser.py**: Markdown parser

### Embedding Layer (`src/embedding/`)
- **base.py**: Base embedding model interface
- **model_loader.py**: Model loading and management
- **cache.py**: Embedding caching for performance
- **models/**: Specific model implementations

### LLM Layer (`src/llm/`)
- **base.py**: Base LLM interface
- **ollama_client.py**: Ollama HTTP API client
- **prompt_builder.py**: Prompt construction with templates
- **templates/**: Prompt templates for different use cases
- **stream_handler.py**: Streaming response handling

### Task Queue Layer (`src/tasks/`) - IMPLEMENTED
- **celery_app.py**: Celery application configuration with Redis broker
- **document_tasks.py**: Celery tasks for document processing (parse, chunk)
- **embedding_tasks.py**: Celery tasks for embedding generation and storage
- **Queue Architecture**: Separate queues for documents (I/O) and embeddings (CPU/GPU)
- **Task Status**: Tracked via Redis result backend
- **API Integration**: Task status endpoints in `src/api/routes/tasks.py`

### Tests (`tests/`)
- **unit/**: Unit tests for individual components
- **integration/**: Integration tests for API endpoints and workflows
- **e2e/**: End-to-end tests for complete user scenarios

### Scripts (`scripts/`)
- Utility scripts for environment setup, initialization, and maintenance

### Deployments (`deployments/`)
- Docker and Kubernetes configurations for deployment

### Documentation (`docs/`)
- Comprehensive documentation covering design, deployment, and development

### Data Directories (`data/`)
- Storage for documents, models, and cache

## Key Files

### `.env.example`
Template for environment variables:
```env
# Application
APP_NAME=RAG System
APP_ENV=development
APP_DEBUG=True
APP_HOST=0.0.0.0
APP_PORT=8000

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_TIMEOUT=30

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=60

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_CACHE_ENABLED=True

# Document Processing
DOCUMENT_MAX_SIZE=10485760  # 10MB
DOCUMENT_CHUNK_SIZE=1000
DOCUMENT_CHUNK_OVERLAP=200
DOCUMENT_SUPPORTED_FORMATS=pdf,docx,txt,md

# Storage
STORAGE_PATH=./data/documents
MODEL_CACHE_PATH=./data/models
CACHE_PATH=./data/cache

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Celery Task Queue
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_TASK_TIME_LIMIT=3600
CELERY_TASK_SOFT_TIME_LIMIT=3000
CELERY_WORKER_PREFETCH_MULTIPLIER=4
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
```

### `requirements.txt`
Core dependencies:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# Qdrant
qdrant-client==1.7.0

# Embeddings
sentence-transformers==2.2.2
torch==2.1.0

# Document Processing
PyPDF2==3.0.1
python-docx==1.1.0
markdown==3.5.0

# Task Queue
celery==5.3.4
redis==5.0.1

# Utilities
httpx==0.25.2
structlog==23.2.0
python-multipart==0.0.6
```

### `requirements-dev.txt`
Development dependencies:
```
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.0
pre-commit==3.5.0
```

## Architecture Principles

1. **Separation of Concerns**: Each layer has a clear responsibility
2. **Dependency Injection**: Services are injected, not instantiated directly
3. **Interface-Based Design**: Base classes define contracts
4. **Error Handling**: Centralized exception handling
5. **Logging**: Structured logging throughout
6. **Testing**: Comprehensive test coverage
7. **Configuration**: Externalized configuration
8. **Documentation**: Inline and external documentation

## Development Workflow

1. **Setup**: Run `scripts/setup_dev_env.sh` to initialize the environment
2. **Start Services**: 
   - Start Redis: `docker-compose up redis -d`
   - Start Celery Worker: `scripts/start_celery_worker.sh` or `docker-compose up celery-worker -d`
   - Start API Server: `uvicorn src.api.main:app --reload`
3. **Development**: Use hot reload with `uvicorn --reload`
4. **Testing**: Run `pytest` for tests
5. **Linting**: Use `black`, `isort`, and `flake8` for code quality
6. **Documentation**: Update docs in the `docs/` directory

### Task Queue Workflow

1. **Upload Document**: `POST /api/v1/documents/upload` → Returns task ID immediately
2. **Check Status**: `GET /api/v1/tasks/{task_id}` → Monitor task progress
3. **Background Processing**: Celery workers process tasks asynchronously
4. **Result**: Task status updates from PENDING → STARTED → SUCCESS

## Deployment Workflow

1. **Build**: `docker build -t rag-system .`
2. **Run Dev**: `docker-compose up`
3. **Run Prod**: `docker-compose -f docker-compose.prod.yml up`
4. **Monitor**: Check logs in `logs/` directory

## Extension Points

1. **New Document Formats**: Add parsers in `src/parsers/`
2. **New Embedding Models**: Add models in `src/embedding/models/`
3. **New LLM Providers**: Implement base interface in `src/llm/`
4. **New Chunking Strategies**: Add strategies in `src/utils/text_chunker.py`
5. **Custom Middleware**: Add middleware in `src/api/middleware/`
6. **New Task Types**: Add Celery tasks in `src/tasks/` and register in `celery_app.py`
7. **Custom Queue Routing**: Configure task routing in `src/tasks/celery_app.py`
8. **Task Monitoring**: Extend task status tracking in `src/api/routes/tasks.py`

This structure provides a solid foundation for building, testing, and deploying a scalable RAG system.
