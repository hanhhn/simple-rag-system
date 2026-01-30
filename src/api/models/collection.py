"""
Collection-related API models.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    """Request model for collection creation."""
    
    name: str = Field(..., min_length=1, max_length=64, description="Collection name")
    dimension: Optional[int] = Field(default=None, ge=1, le=10000, description="Embedding dimension")
    distance_metric: str = Field(
        default="Cosine",
        description="Distance metric (Cosine, Euclid, Dot)"
    )


class CollectionInfo(BaseModel):
    """Collection information model."""
    
    name: str = Field(description="Collection name")
    vector_count: int = Field(description="Number of vectors in collection")
    dimension: int = Field(description="Vector dimension")
    status: str = Field(description="Collection status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    distance_metric: str = Field(description="Distance metric used")


class CollectionListResponse(BaseModel):
    """Response model for collection list."""
    
    collections: list[CollectionInfo] = Field(description="List of collections")
    total: int = Field(description="Total number of collections")


class CollectionResponse(BaseModel):
    """Response model for collection operations."""
    
    collection: CollectionInfo = Field(description="Collection information")
    message: str = Field(description="Operation message")
