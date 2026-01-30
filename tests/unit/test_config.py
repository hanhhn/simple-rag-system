"""Tests for Configuration Module

Unit tests for configuration management.
"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_settings_default_values():
    """Test that settings have default values."""
    settings = Settings()
    assert settings.APP_NAME == "RAG System"
    assert settings.APP_ENV == "development"
    assert settings.APP_DEBUG is True
    assert settings.APP_HOST == "0.0.0.0"
    assert settings.APP_PORT == 8000


def test_settings_from_env_file():
    """Test loading settings from .env file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("APP_NAME=Test App\n")
        f.write("APP_PORT=9000\n")
        env_path = Path(f.name)

    try:
        settings = Settings(_env_file=env_path)
        assert settings.APP_NAME == "Test App"
        assert settings.APP_PORT == 9000
    finally:
        env_path.unlink()


def test_settings_supported_formats():
    """Test that supported_formats property returns list."""
    settings = Settings(DOCUMENT_SUPPORTED_FORMATS="pdf,docx,txt,md")
    formats = settings.supported_formats
    assert isinstance(formats, list)
    assert len(formats) == 4
    assert "pdf" in formats
    assert "docx" in formats
    assert "txt" in formats
    assert "md" in formats


def test_settings_caching():
    """Test that settings are cached."""
    settings1 = Settings()
    settings2 = Settings()
    assert settings1 is settings2


def test_qdrant_url():
    """Test Qdrant URL configuration."""
    settings = Settings(QDRANT_URL="http://localhost:6333")
    assert settings.QDRANT_URL == "http://localhost:6333"


def test_ollama_configuration():
    """Test Ollama configuration."""
    settings = Settings(OLLAMA_URL="http://localhost:11434", OLLAMA_MODEL="llama2")
    assert settings.OLLAMA_URL == "http://localhost:11434"
    assert settings.OLLAMA_MODEL == "llama2"


def test_document_max_size():
    """Test document max size configuration."""
    settings = Settings(DOCUMENT_MAX_SIZE=10485760)
    assert settings.DOCUMENT_MAX_SIZE == 10485760


def test_chunking_configuration():
    """Test chunking configuration."""
    settings = Settings(DOCUMENT_CHUNK_SIZE=1000, DOCUMENT_CHUNK_OVERLAP=200)
    assert settings.DOCUMENT_CHUNK_SIZE == 1000
    assert settings.DOCUMENT_CHUNK_OVERLAP == 200
