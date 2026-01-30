"""
Query-related API models.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query."""
    
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    collection: str = Field(..., min_length=1, max_length=64, description="Collection name")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    use_rag: bool = Field(default=True, description="Whether to use RAG generation")
    stream: bool = Field(default=False, description="Whether to stream response")


class RetrievedDocument(BaseModel):
    """Retrieved document model."""
    
    id: str = Field(description="Document ID")
    score: float = Field(description="Similarity score")
    text: str = Field(description="Document text content")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for query."""
    
    query: str = Field(description="Original query")
    answer: Optional[str] = Field(description="Generated answer (if use_rag=True)")
    answer_chunks: Optional[list[str]] = Field(description="Streaming answer chunks (if stream=True)")
    retrieved_documents: list[RetrievedDocument] = Field(description="Retrieved documents")
    retrieval_count: int = Field(description="Number of documents retrieved")
    collection: str = Field(description="Collection searched")
    top_k: int = Field(description="Number of documents requested")
    score_threshold: float = Field(description="Score threshold used")
    use_rag: bool = Field(description="Whether RAG was used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")
