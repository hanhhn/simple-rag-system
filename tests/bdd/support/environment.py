"""
Before and after hooks for BDD tests.
"""
import asyncio
from typing import Generator

from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.main import app


def before_all(context):
    """
    Run once before all features.
    Initialize shared resources.
    """
    context.base_url = "http://localhost:8000/api/v1"
    context.test_client = TestClient(app)


def after_all(context):
    """
    Run once after all features.
    Cleanup resources.
    """
    pass


def before_scenario(context, scenario):
    """
    Run before each scenario.
    Reset context state.
    """
    # Reset scenario-specific context
    context.response = None
    context.status_code = None
    context.response_data = None
    context.document_id = None
    context.collection_name = None
    context.batch_id = None
    context.version_id = None
    context.error_message = None

    # Store scenario name for logging
    context.scenario_name = scenario.name


def after_scenario(context, scenario):
    """
    Run after each scenario.
    Cleanup scenario-specific resources.
    """
    # Cleanup test data if needed
    if hasattr(context, 'created_collections'):
        for collection_name in context.created_collections:
            try:
                cleanup_collection(context, collection_name)
            except Exception as e:
                print(f"Failed to cleanup collection {collection_name}: {e}")


def cleanup_collection(context, collection_name: str):
    """
    Helper to cleanup a test collection.
    """
    try:
        response = context.test_client.delete(
            f"{context.base_url}/collections/{collection_name}"
        )
        if response.status_code == 200 or response.status_code == 404:
            print(f"Cleaned up collection: {collection_name}")
    except Exception as e:
        print(f"Error cleaning up collection {collection_name}: {e}")


def before_feature(context, feature):
    """
    Run before each feature.
    """
    pass


def after_feature(context, feature):
    """
    Run after each feature.
    """
    pass


def before_step(context, step):
    """
    Run before each step.
    """
    pass


def after_step(context, step):
    """
    Run after each step.
    Log step results.
    """
    if step.status == "failed":
        print(f"Step failed: {step}")
        if hasattr(context, 'response') and context.response:
            print(f"Response status: {context.status_code}")
            print(f"Response data: {context.response_data}")
