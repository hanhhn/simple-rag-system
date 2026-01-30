# Hướng dẫn Setup với Conda

Tài liệu này hướng dẫn chi tiết cách setup và chạy ứng dụng RAG System sử dụng Conda.

## Yêu cầu

- Conda 4.6+ hoặc Miniconda/Anaconda đã được cài đặt
- Python 3.11+ (sẽ được cài tự động qua environment.yml)

## Bước 1: Tạo Conda Environment

### Tạo môi trường từ file environment.yml

```bash
# Tạo môi trường với dependencies cơ bản
conda env create -f environment.yml

# Hoặc tạo môi trường với đầy đủ dev dependencies
conda env create -f environment-dev.yml
```

### Kích hoạt môi trường

```bash
# Kích hoạt môi trường
conda activate simple-rag-system

# Hoặc nếu dùng environment-dev.yml
conda activate simple-rag-system-dev
```

## Bước 2: Cấu hình Environment Variables

Đảm bảo file `.env` đã được tạo:

```bash
# Tự động tạo từ env.example
copy env.example .env  # Windows
cp env.example .env    # Linux/Mac

# Hoặc sử dụng script
scripts\setup_env.bat  # Windows
./scripts/setup_env.sh # Linux/Mac
```

Kiểm tra và chỉnh sửa các biến môi trường trong `.env` nếu cần.

## Bước 3: Khởi động Services

### Sử dụng Docker Compose

```bash
# Khởi động Qdrant, Ollama, Redis
docker-compose up -d qdrant ollama redis

# Kiểm tra status
docker-compose ps
```

## Bước 4: Chạy Ứng dụng

### Option A: Sử dụng Script Tự động (Khuyến nghị)

Script sẽ tự động:
- Kiểm tra conda có sẵn
- Tạo môi trường nếu chưa có
- Kích hoạt môi trường
- Tạo file .env nếu chưa có
- Khởi động ứng dụng

**Windows:**
```bash
scripts\start_with_conda.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start_with_conda.sh
./scripts/start_with_conda.sh
```

### Option B: Chạy Thủ công

```bash
# Kích hoạt môi trường
conda activate simple-rag-system

# Chạy ứng dụng
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Bước 5: Chạy Celery Worker (Nếu cần)

Script Celery worker đã được cập nhật để tự động phát hiện và sử dụng conda environment:

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
conda activate simple-rag-system
celery -A src.tasks.celery_app worker --loglevel=info --queues=documents,embeddings --concurrency=4
```

## Quản lý Conda Environment

### Xem danh sách environments

```bash
conda env list
```

### Cập nhật environment

Khi có thay đổi trong `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Cài thêm packages

```bash
conda activate simple-rag-system
pip install package-name
```

### Cài thêm dev dependencies

Nếu bạn đã tạo môi trường từ `environment.yml` và muốn cài thêm dev dependencies:

```bash
conda activate simple-rag-system
pip install -r requirements-dev.txt
```

### Xóa environment

```bash
# Deactivate trước
conda deactivate

# Xóa environment
conda env remove -n simple-rag-system
```

### Export environment

Để export environment hiện tại:

```bash
conda env export > environment-custom.yml
```

## Troubleshooting

### Lỗi: Conda command not found

- Đảm bảo conda đã được cài đặt và thêm vào PATH
- Trên Windows, có thể cần mở lại terminal sau khi cài conda
- Kiểm tra: `conda --version`

### Lỗi: Environment not found

```bash
# Kiểm tra environment có tồn tại
conda env list

# Tạo lại nếu cần
conda env create -f environment.yml
```

### Lỗi: Package conflicts

```bash
# Cập nhật conda
conda update conda

# Tạo lại environment
conda env remove -n simple-rag-system
conda env create -f environment.yml
```

### Lỗi: Cannot activate environment

Trên Windows, có thể cần chạy:
```bash
conda init cmd.exe
# Hoặc
conda init powershell
```

Sau đó đóng và mở lại terminal.

### Lỗi: Python version mismatch

Đảm bảo `environment.yml` chỉ định đúng Python version (3.11). Nếu cần thay đổi:

```bash
# Sửa environment.yml, thay đổi python=3.11 thành version bạn muốn
# Sau đó tạo lại environment
conda env remove -n simple-rag-system
conda env create -f environment.yml
```

## Lợi ích của Conda

1. **Quản lý dependencies tốt hơn**: Conda quản lý cả Python packages và system dependencies
2. **Isolation tốt**: Mỗi environment hoàn toàn độc lập
3. **Dễ dàng chia sẻ**: File `environment.yml` có thể được commit vào git
4. **Hỗ trợ nhiều Python versions**: Dễ dàng switch giữa các Python versions
5. **Binary packages**: Nhiều packages được cung cấp dưới dạng binary, cài đặt nhanh hơn

## So sánh với venv

| Tính năng | Conda | venv |
|-----------|-------|------|
| Quản lý Python version | ✅ | ❌ |
| Quản lý system dependencies | ✅ | ❌ |
| Binary packages | ✅ | ❌ (chủ yếu source) |
| Kích thước | Lớn hơn | Nhỏ hơn |
| Tốc độ cài đặt | Nhanh hơn (binary) | Chậm hơn (compile) |
| Phù hợp cho | Data science, ML | Web development đơn giản |

## Tài liệu tham khảo

- [Conda Documentation](https://docs.conda.io/)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Environment Files](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually)
