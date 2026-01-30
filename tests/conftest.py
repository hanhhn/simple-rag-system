"""Pytest Configuration

Shared fixtures and configuration for pytest.
"""

import pytest

from src.core.config import Settings


@pytest.fixture
def settings():
    """
    Fixture that provides a Settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


@pytest.fixture
def sample_document():
    """
    Fixture that provides a sample document.

    Returns:
        dict: Sample document data
    """
    return {
        "id": "doc-123",
        "filename": "test.pdf",
        "content": "This is a test document content.",
        "metadata": {
            "author": "Test Author",
            "created_at": "2024-01-01",
        },
    }


@pytest.fixture
def sample_query():
    """
    Fixture that provides a sample query.

    Returns:
        dict: Sample query data
    """
    return {
        "query": "What is this document about?",
        "collection": "test_collection",
        "top_k": 5,
    }
