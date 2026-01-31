# Quick Start - Local Development

## Bước 1: Tạo file .env

File `.env` đã được tạo tự động. Nếu chưa có, chạy:

```bash
# Windows
copy env.example .env

# Linux/Mac  
cp env.example .env
```

## Bước 2: Khởi động các services cần thiết

### Option A: Docker Compose (Khuyến nghị)

```bash
# Khởi động Qdrant, Ollama, Redis
docker-compose up -d qdrant ollama redis celery-worker

# Kiểm tra status
docker-compose ps
```

### Option B: Chạy riêng lẻ

```bash
# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest

# Ollama (model sẽ được tự động pull khi start với docker-compose)
# Nếu chạy riêng, bạn cần pull model thủ công:
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
docker exec ollama ollama pull llama2

# Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

**Lưu ý:** Khi sử dụng Docker Compose, Ollama sẽ tự động pull model được cấu hình trong `OLLAMA_MODEL` khi khởi động. Xem [docs/14-ollama-model-management.md](docs/14-ollama-model-management.md) để biết chi tiết.

## Bước 3: Cài đặt Python dependencies

### Option A: Sử dụng Conda (Khuyến nghị nếu bạn dùng conda)

```bash
# Tạo môi trường conda từ file environment.yml
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate simple-rag-system

# Hoặc nếu muốn cài thêm dev dependencies
conda env create -f environment-dev.yml
conda activate simple-rag-system-dev
```

**Lưu ý:** Nếu bạn muốn cài thêm dev dependencies sau khi đã tạo môi trường:
```bash
conda activate simple-rag-system
pip install -r requirements-dev.txt
```

### Option B: Sử dụng venv (Python virtual environment)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Bước 4: Chạy ứng dụng

### Backend API

**Với Conda:**
```bash
# Windows
scripts\start_with_conda.bat

# Linux/Mac
chmod +x scripts/start_with_conda.sh
./scripts/start_with_conda.sh

# Hoặc thủ công
conda activate simple-rag-system
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Với venv:**
```bash
# Kích hoạt environment trước
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Chạy ứng dụng
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Celery Worker (nếu cần)

```bash
# Windows
scripts\start_celery_worker.bat

# Linux/Mac
./scripts/start_celery_worker.sh
```

### Frontend (nếu cần)

```bash
cd frontend
npm install
npm run dev
```

## Bước 5: Kiểm tra

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Các biến môi trường quan trọng

Kiểm tra file `.env` và đảm bảo:

- `QDRANT_URL=http://localhost:6333` (hoặc `http://qdrant:6333` nếu dùng Docker network)
- `OLLAMA_URL=http://localhost:11434` (hoặc `http://ollama:11434`)
- `CELERY_BROKER_URL=redis://localhost:6379/0` (hoặc `redis://redis:6379/0`)
- `OLLAMA_MODEL=llama2` (hoặc model bạn muốn dùng)

## Troubleshooting

### Lỗi kết nối Qdrant
```bash
# Kiểm tra Qdrant đang chạy
docker ps | grep qdrant
curl http://localhost:6333/health
```

### Lỗi kết nối Ollama
```bash
# Kiểm tra Ollama và model
docker ps | grep ollama
docker exec ollama ollama list
```

### Lỗi import modules
```bash
# Đảm bảo đã activate virtual environment hoặc conda environment
# Với venv:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Với conda:
conda activate simple-rag-system

# Chạy từ thư mục gốc
python -m src.api.main
```

### Lỗi: NumPy version incompatibility

Nếu gặp lỗi `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:

```bash
# Với conda:
conda activate simple-rag-system
conda install "numpy<2" -y

# Với venv:
pip install "numpy<2" --upgrade
```

**Lưu ý:** File `requirements.txt` và `environment.yml` đã được cấu hình với `numpy<2` để tránh lỗi này.

Xem [LOCAL_SETUP.md](LOCAL_SETUP.md) để biết chi tiết hơn.
