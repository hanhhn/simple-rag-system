#!/bin/bash
# Start Celery worker for background task processing

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment variables if not set
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/0}"

# Start Celery worker
echo "Starting Celery worker..."
echo "Broker: $CELERY_BROKER_URL"
echo "Backend: $CELERY_RESULT_BACKEND"

celery -A src.tasks.celery_app worker \
    --loglevel=info \
    --queues=documents,embeddings \
    --concurrency=4 \
    --max-tasks-per-child=1000 \
    --time-limit=3600 \
    --soft-time-limit=3000
