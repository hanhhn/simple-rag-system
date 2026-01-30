"""
Document-related API models.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    collection: str = Field(..., min_length=1, max_length=64, description="Collection name")
    chunk_size: Optional[int] = Field(default=None, ge=100, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, description="Chunk overlap in characters")
    chunker_type: Optional[str] = Field(
        default="sentence",
        description="Chunker type (character, word, sentence, paragraph, recursive)"
    )
    metadata: Optional[dict] = Field(default=None, description="Additional metadata for document")


class DocumentChunk(BaseModel):
    """Document chunk model."""
    
    id: str = Field(description="Chunk ID")
    text: str = Field(description="Chunk text content")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    
    document_id: str = Field(description="Document ID")
    filename: str = Field(description="Original filename")
    collection: str = Field(description="Collection name")
    chunk_count: int = Field(description="Number of chunks created")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    
    documents: list[DocumentResponse] = Field(description="List of documents")
    total: int = Field(description="Total number of documents")
    collection: str = Field(description="Collection name")


class DocumentChunkResponse(BaseModel):
    """Response model for document chunks."""
    
    chunks: list[DocumentChunk] = Field(description="List of chunks")
    document_id: str = Field(description="Document ID")
    total_chunks: int = Field(description="Total number of chunks")


class DocumentUploadTaskResponse(BaseModel):
    """Response model for document upload task (async)."""
    
    task_id: str = Field(description="Document processing task ID")
    document_id: str = Field(description="Document ID")
    filename: str = Field(description="Original filename")
    collection: str = Field(description="Collection name")
    status: str = Field(description="Task status (PENDING, STARTED, SUCCESS, FAILURE)")
    message: str = Field(description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")