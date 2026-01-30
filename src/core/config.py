"""
Configuration management for the RAG system.

This module provides centralized configuration management using Pydantic Settings.
All configuration is loaded from environment variables with sensible defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration settings."""

    app_name: str = Field(default="RAG System", description="Application name")
    app_env: Literal["development", "testing", "production"] = Field(
        default="development", description="Application environment"
    )
    app_debug: bool = Field(default=True, description="Debug mode")
    app_host: str = Field(default="0.0.0.0", description="API host")
    app_port: int = Field(default=8000, description="API port")
    api_version: str = Field(default="v1", description="API version")
    
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    @field_validator("app_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: str = Field(default="", description="Qdrant API key (optional)")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")


class OllamaConfig(BaseSettings):
    """Ollama LLM configuration."""

    url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="llama2", description="Default LLM model")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=2000, ge=1, description="Maximum response tokens")
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name or path"
    )
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding generation")
    cache_enabled: bool = Field(default=True, description="Enable embedding cache")
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="Device for model inference")
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")


class DocumentConfig(BaseSettings):
    """Document processing configuration."""

    max_size: int = Field(default=10485760, description="Maximum document size in bytes (10MB)")
    chunk_size: int = Field(default=1000, ge=100, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    supported_formats: list[str] = Field(
        default=["pdf", "docx", "txt", "md"],
        description="Supported document formats"
    )
    
    model_config = SettingsConfigDict(env_prefix="DOCUMENT_", env_file=".env", extra="ignore")

    @field_validator("supported_formats", mode="before")
    @classmethod
    def parse_formats(cls, v: str | list[str]) -> list[str]:
        """Parse supported formats from comma-separated string or list."""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",")]
        return [fmt.lower() for fmt in v]


class StorageConfig(BaseSettings):
    """Storage configuration."""

    storage_path: Path = Field(
        default=Path("./data/documents"),
        description="Path to document storage directory"
    )
    model_cache_path: Path = Field(
        default=Path("./data/models"),
        description="Path to model cache directory"
    )
    cache_path: Path = Field(
        default=Path("./data/cache"),
        description="Path to general cache directory"
    )
    log_path: Path = Field(
        default=Path("./logs"),
        description="Path to log directory"
    )
    
    model_config = SettingsConfigDict(env_prefix="STORAGE_", env_file=".env", extra="ignore")

    def model_post_init(self, __context: object) -> None:
        """Create directories if they don't exist."""
        for path in [self.storage_path, self.model_cache_path, self.cache_path, self.log_path]:
            path.mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    
    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=".env", extra="ignore")


class SecurityConfig(BaseSettings):
    """Security configuration."""

    jwt_secret_key: str = Field(
        default="change-this-secret-key-in-production",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT token expiration in hours")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    
    model_config = SettingsConfigDict(env_prefix="JWT_" if key.startswith("jwt") else "", env_file=".env", extra="ignore")


class CeleryConfig(BaseSettings):
    """Celery task queue configuration."""

    broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis)"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL (Redis)"
    )
    task_serializer: str = Field(default="json", description="Task serialization format")
    result_serializer: str = Field(default="json", description="Result serialization format")
    accept_content: list[str] = Field(
        default=["json"],
        description="Accepted content types"
    )
    timezone: str = Field(default="UTC", description="Timezone for task scheduling")
    enable_utc: bool = Field(default=True, description="Enable UTC timezone")
    task_track_started: bool = Field(default=True, description="Track task start time")
    task_time_limit: int = Field(default=3600, ge=1, description="Task time limit in seconds")
    task_soft_time_limit: int = Field(default=3000, ge=1, description="Task soft time limit in seconds")
    worker_prefetch_multiplier: int = Field(default=4, ge=1, description="Worker prefetch multiplier")
    worker_max_tasks_per_child: int = Field(default=1000, ge=1, description="Max tasks per worker child")
    
    model_config = SettingsConfigDict(env_prefix="CELERY_", env_file=".env", extra="ignore")


class Config:
    """
    Main configuration class that aggregates all sub-configurations.
    
    This class provides a single point of access to all configuration settings
    throughout the application. It uses the singleton pattern via lru_cache.
    """

    def __init__(self) -> None:
        """Initialize all configuration sections."""
        self.app = AppConfig()
        self.qdrant = QdrantConfig()
        self.ollama = OllamaConfig()
        self.embedding = EmbeddingConfig()
        self.document = DocumentConfig()
        self.storage = StorageConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.celery = CeleryConfig()

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app.app_env == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.app.app_env == "testing"


@lru_cache()
def get_config() -> Config:
    """
    Get the singleton configuration instance.
    
    Returns:
        Config: The cached configuration instance.
    
    Example:
        >>> config = get_config()
        >>> print(config.app.app_name)
    """
    return Config()
