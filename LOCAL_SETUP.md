# Hướng dẫn Setup Local Development

Tài liệu này hướng dẫn cách cấu hình môi trường để chạy và debug hệ thống RAG local.

## Bước 1: Tạo file .env

Có 3 cách để tạo file `.env`:

### Cách 1: Sử dụng script tự động (Khuyến nghị)

**Windows:**
```bash
scripts\setup_env.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

**Python (mọi hệ điều hành):**
```bash
python scripts/setup_env.py
```

### Cách 2: Copy thủ công
```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

### Cách 3: Tạo file .env mới
Tạo file `.env` trong thư mục gốc và copy nội dung từ `env.example`.

## Bước 2: Kiểm tra và chỉnh sửa file .env

Mở file `.env` và kiểm tra các cấu hình sau:

### Cấu hình cơ bản (quan trọng nhất)

1. **Qdrant URL**: 
   - Nếu chạy local (không dùng Docker): `QDRANT_URL=http://localhost:6333`
   - Nếu dùng Docker Compose: `QDRANT_URL=http://qdrant:6333` (hoặc giữ `localhost:6333` nếu expose port)

2. **Ollama URL**:
   - Nếu chạy local: `OLLAMA_URL=http://localhost:11434`
   - Nếu dùng Docker Compose: `OLLAMA_URL=http://ollama:11434` (hoặc giữ `localhost:11434`)

3. **Redis/Celery**:
   - Nếu chạy local: `CELERY_BROKER_URL=redis://localhost:6379/0`
   - Nếu dùng Docker Compose: `CELERY_BROKER_URL=redis://redis:6379/0` (hoặc giữ `localhost:6379`)

### Các cấu hình khác

- **OLLAMA_MODEL**: Model LLM bạn muốn sử dụng (mặc định: `llama2`)
- **EMBEDDING_DEVICE**: `cpu` hoặc `cuda` (nếu có GPU)
- **LOG_LEVEL**: `DEBUG` để debug chi tiết, `INFO` cho thông thường

## Bước 3: Khởi động các services cần thiết

### Option A: Sử dụng Docker Compose (Khuyến nghị)

```bash
# Khởi động tất cả services (Qdrant, Ollama, Redis)
docker-compose up -d qdrant ollama redis

# Kiểm tra logs
docker-compose logs -f
```

### Option B: Chạy từng service riêng lẻ

**Qdrant:**
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

**Ollama:**
```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest

# Pull model (ví dụ: llama2)
docker exec ollama ollama pull llama2
```

**Redis:**
```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

## Bước 4: Cài đặt dependencies Python

### Option A: Sử dụng Conda (Khuyến nghị nếu bạn dùng conda)

```bash
# Tạo môi trường conda từ file environment.yml
conda env create -f environment.yml

# Kích hoạt môi trường
conda activate simple-rag-system

# Hoặc nếu muốn cài thêm dev dependencies ngay từ đầu
conda env create -f environment-dev.yml
conda activate simple-rag-system-dev
```

**Lưu ý:** 
- Nếu bạn đã tạo môi trường từ `environment.yml` và muốn cài thêm dev dependencies sau:
  ```bash
  conda activate simple-rag-system
  pip install -r requirements-dev.txt
  ```

- Để xóa môi trường conda (nếu cần):
  ```bash
  conda deactivate
  conda env remove -n simple-rag-system
  ```

- Để cập nhật môi trường khi có thay đổi trong `environment.yml`:
  ```bash
  conda env update -f environment.yml --prune
  ```

### Option B: Sử dụng venv (Python virtual environment)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Bước 5: Chạy ứng dụng

### Chạy Backend API

**Với Conda (Khuyến nghị nếu bạn dùng conda):**

```bash
# Sử dụng script tự động (tự động kiểm tra và tạo environment nếu cần)
# Windows:
scripts\start_with_conda.bat

# Linux/Mac:
chmod +x scripts/start_with_conda.sh
./scripts/start_with_conda.sh

# Hoặc thủ công:
conda activate simple-rag-system
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Với venv:**

```bash
# Đảm bảo đã kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Chạy ứng dụng
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Hoặc chạy trực tiếp:

```bash
python -m src.api.main
```

### Chạy Celery Worker (nếu cần xử lý tasks)

**Windows:**
```bash
scripts\start_celery_worker.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start_celery_worker.sh
./scripts/start_celery_worker.sh
```

Hoặc thủ công:
```bash
celery -A src.tasks.celery_app worker --loglevel=info --queues=documents,embeddings --concurrency=4
```

### Chạy Frontend (nếu cần)

```bash
cd frontend
npm install
npm run dev
```

Frontend sẽ chạy tại: http://localhost:5173

## Bước 6: Kiểm tra kết nối

1. **API Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **API Documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Qdrant Dashboard:**
   - http://localhost:6333/dashboard

## Troubleshooting

### Lỗi kết nối Qdrant

- Kiểm tra Qdrant đã chạy: `docker ps | grep qdrant`
- Kiểm tra port 6333: `curl http://localhost:6333/health`
- Kiểm tra `QDRANT_URL` trong `.env`

### Lỗi kết nối Ollama

- Kiểm tra Ollama đã chạy: `docker ps | grep ollama`
- Kiểm tra model đã được pull: `docker exec ollama ollama list`
- Kiểm tra `OLLAMA_URL` và `OLLAMA_MODEL` trong `.env`

### Lỗi kết nối Redis

- Kiểm tra Redis đã chạy: `docker ps | grep redis`
- Test kết nối: `docker exec redis redis-cli ping`
- Kiểm tra `CELERY_BROKER_URL` trong `.env`

### Lỗi import modules

- Đảm bảo đã activate virtual environment hoặc conda environment
  - Với venv: `venv\Scripts\activate` (Windows) hoặc `source venv/bin/activate` (Linux/Mac)
  - Với conda: `conda activate simple-rag-system`
- Kiểm tra PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` (Linux/Mac)
- Hoặc chạy từ thư mục gốc với: `python -m src.api.main`
- Kiểm tra Python version: `python --version` (cần Python 3.11+)

### Lỗi: NumPy version incompatibility

Nếu gặp lỗi `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` khi import torch hoặc sentence-transformers:

**Nguyên nhân:** `torch` và `sentence-transformers` được biên dịch với NumPy 1.x, nhưng môi trường đang có NumPy 2.x.

**Giải pháp:**

**Với Conda:**
```bash
conda activate simple-rag-system
conda install "numpy<2" -y
# Hoặc nếu conda không thay đổi được version
pip install "numpy<2" --upgrade
```

**Với venv:**
```bash
# Kích hoạt virtual environment
source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows

# Downgrade NumPy
pip install "numpy<2" --upgrade
```

**Lưu ý:** File `requirements.txt`, `environment.yml` và `environment-dev.yml` đã được cấu hình với `numpy<2` để tránh lỗi này. Nếu bạn cài đặt từ các file này, lỗi sẽ không xảy ra.

### Lỗi: KeyError 'modernbert'

Nếu gặp lỗi `KeyError: 'modernbert'` khi chạy ứng dụng với Granite embedding model:

**Nguyên nhân:** Phiên bản `transformers` cũ không hỗ trợ kiến trúc ModernBERT mà Granite model sử dụng.

**Giải pháp:**

**Với Conda:**
```bash
conda activate simple-rag-system
conda env update -f environment.yml --prune
```

Hoặc cài thủ công:
```bash
conda activate simple-rag-system
pip install --upgrade transformers sentence-transformers torch
```

**Với venv:**
```bash
source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows
pip install --upgrade transformers sentence-transformers torch
```

**Phiên bản tối thiểu yêu cầu:**
- `transformers>=5.0.0`
- `sentence-transformers>=5.2.2`
- `torch>=2.10.0`

**Lưu ý:** Các file `requirements.txt`, `environment.yml` và `environment-dev.yml` đã được cập nhật với các phiên bản mới này. Xem [GRANITE_MIGRATION.md](GRANITE_MIGRATION.md) để biết thêm chi tiết.

### Debug mode

Để bật debug mode, set trong `.env`:
```
APP_DEBUG=True
LOG_LEVEL=DEBUG
```

## Cấu trúc file .env

File `.env` được chia thành các section:

- **Application**: Cấu hình app (host, port, debug mode)
- **Qdrant**: Cấu hình vector database
- **Ollama**: Cấu hình LLM
- **Embedding**: Cấu hình embedding model
- **Document**: Cấu hình xử lý documents
- **Storage**: Đường dẫn lưu trữ
- **Logging**: Cấu hình logging
- **Security**: JWT và rate limiting
- **Celery**: Cấu hình task queue

Xem file `env.example` để biết chi tiết về từng biến môi trường.

## Lưu ý

- File `.env` không được commit vào git (đã có trong `.gitignore`)
- File `env.example` là template, có thể commit
- Khi chạy với Docker Compose, một số biến môi trường có thể được override trong `docker-compose.yml`
- Để chạy production, sử dụng `docker-compose.prod.yml` và cấu hình phù hợp
