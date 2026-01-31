"""
Celery application configuration.

This module sets up the Celery application for background task processing.
"""
from celery import Celery
from celery.signals import setup_logging, worker_process_init

from src.core.config import get_config
from src.core.logging import configure_logging, get_logger


# Configure logging before creating Celery app
configure_logging()
logger = get_logger(__name__)

# Get configuration
config = get_config()

# Create Celery app
celery_app = Celery(
    "rag_system",
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer=config.celery.task_serializer,
    result_serializer=config.celery.result_serializer,
    accept_content=config.celery.accept_content,
    timezone=config.celery.timezone,
    enable_utc=config.celery.enable_utc,
    task_track_started=config.celery.task_track_started,
    task_time_limit=config.celery.task_time_limit,
    task_soft_time_limit=config.celery.task_soft_time_limit,
    worker_prefetch_multiplier=config.celery.worker_prefetch_multiplier,
    worker_max_tasks_per_child=config.celery.worker_max_tasks_per_child,
    worker_pool=config.celery.worker_pool,
    # Task routing
    task_routes={
        "src.tasks.document_tasks.*": {"queue": "documents"},
        "src.tasks.embedding_tasks.*": {"queue": "embeddings"},
    },
    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    # Worker settings
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    # Task discovery
    include=["src.tasks.document_tasks", "src.tasks.embedding_tasks"],
)

# Disable Celery's default logging setup to use our structured logging
@setup_logging.connect
def config_loggers(*args, **kwargs):
    """Configure logging for Celery workers."""
    configure_logging()
    logger.info("Celery logging configured")


@worker_process_init.connect
def init_worker_process(*args, **kwargs):
    """
    Initialize worker process after fork.
    
    This signal is sent after a worker process has been forked.
    For prefork pools, this is where we can safely initialize
    resources that don't work well with fork (like PyTorch models).
    
    Note: For thread/solo pools, this may not be called or may
    be called differently, so we use lazy loading in services.
    """
    logger.info(
        "Worker process initialized",
        worker_pool=config.celery.worker_pool
    )
    
    # Clear any cached models to force reload after fork
    # This helps avoid issues with models loaded before fork
    try:
        from src.embedding.model_loader import ModelLoader
        # The ModelLoader uses a class-level cache, but each worker
        # process will have its own memory space, so this is safe
        logger.info("Worker process ready for model loading")
    except Exception as e:
        logger.warning(
            "Error during worker process initialization",
            error=str(e)
        )


logger.info(
    "Celery app initialized",
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
    worker_pool=config.celery.worker_pool,
)
