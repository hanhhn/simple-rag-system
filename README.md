# Simple RAG System

A Retrieval-Augmented Generation (RAG) system that combines large language models with vector-based information retrieval. The system enables users to upload documents, process them into embeddings, and ask natural language questions that are answered based on the document content.

## Features

- 📄 **Document Ingestion**: Support for PDF, TXT, MD, and DOCX files
- 🔍 **Vector Search**: High-performance similarity search using Qdrant
- 🤖 **Local LLM**: Privacy-focused responses using local LLM models via Ollama
- 🚀 **REST API**: Clean and well-documented API endpoints
- 🎯 **Multiple Collections**: Support for multiple document collections
- 📊 **Monitoring**: Built-in metrics and monitoring support

## Architecture

The system follows a layered architecture:

- **API Layer**: FastAPI-based REST API
- **Service Layer**: Business logic and orchestration
- **Data Layer**: Vector database (Qdrant) and file storage
- **Infrastructure Layer**: Configuration, logging, and monitoring

## Tech Stack

- **Backend**: Python 3.11+, FastAPI
- **Vector Database**: Qdrant
- **LLM Runtime**: Ollama (Llama 2, Mistral, etc.)
- **Embeddings**: sentence-transformers
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/simple-rag-system.git
cd simple-rag-system
```

2. **Set up environment variables**:
```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env

# Or use the setup script
scripts\setup_env.bat  # Windows
./scripts/setup_env.sh  # Linux/Mac

# Edit .env with your configuration
```

3. **Start services with Docker Compose**:
```bash
docker-compose -f deployments/docker/docker-compose.yml up -d
```

4. **Wait for services to be ready**:
```bash
docker-compose logs -f rag-app
```

5. **Access the API**:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### Manual Installation (Development)

#### Option A: Using Conda (Recommended for Data Science/ML workflows)

1. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate simple-rag-system
```

2. **Install dev dependencies (optional)**:
```bash
pip install -r requirements-dev.txt
```

See [CONDA_SETUP.md](CONDA_SETUP.md) for detailed conda setup instructions.

#### Option B: Using venv (Python virtual environment)

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Start Qdrant**:
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

4. **Start Ollama**:
```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
```

5. **Pull a model**:
```bash
docker exec ollama ollama pull llama2
```

6. **Run the application**:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "collection=my_collection"
```

### Query the System

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "collection": "my_collection",
    "top_k": 5
  }'
```

### List Collections

```bash
curl -X GET "http://localhost:8000/api/v1/collections"
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
simple-rag-system/
├── src/                    # Source code
│   ├── api/               # API layer
│   ├── services/          # Business logic
│   ├── core/              # Core functionality
│   ├── utils/             # Utilities
│   ├── parsers/           # Document parsers
│   ├── embedding/         # Embedding models
│   └── llm/               # LLM integration
├── tests/                 # Test suite
├── deployments/           # Docker configurations
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

For detailed project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_document_processor.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Deployment

### Production Deployment

```bash
docker-compose -f deployments/docker/docker-compose.prod.yml up -d
```

### Environment Variables

See `env.example` for all available configuration options. For detailed setup instructions:
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [LOCAL_SETUP.md](LOCAL_SETUP.md) - Detailed local setup
- [CONDA_SETUP.md](CONDA_SETUP.md) - Conda-specific setup guide

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

To enable monitoring:

```bash
docker-compose -f deployments/docker/docker-compose.yml --profile monitoring up -d
```

## Documentation

- [Basic Design](docs/01-basic-design.md) - System overview
- [C4 Model](docs/02-c4-model.md) - Architecture diagrams
- [High-Level Design](docs/03-high-level-design.md) - Architectural patterns
- [Data Flow](docs/04-data-flow.md) - Data flow diagrams
- [Sequence Diagrams](docs/05-sequence-diagrams.md) - Interaction sequences

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search engine
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [sentence-transformers](https://www.sbert.net/) - Sentence embeddings
- [LangChain](https://langchain.com/) - Framework for LLM applications

## Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## Roadmap

- [ ] Web UI for easier document management
- [ ] Chat history and conversation memory
- [ ] Multi-modal support (images, audio)
- [ ] Advanced chunking strategies
- [ ] Reranking models
- [ ] Multi-language support
- [ ] Fine-tuning capabilities

---

**Note**: This is a simple RAG system designed for demonstration and learning purposes. For production use, consider additional security measures, monitoring, and optimization.
