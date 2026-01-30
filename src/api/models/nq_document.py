"""
Natural Questions (NQ) API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, HttpUrl


class NQQuestionItem(BaseModel):
    """Model for a single question from Natural Questions dataset."""
    
    question: str = Field(
        ...,
        description="The natural language question",
        examples=["What is the capital of France?"]
    )
    answer: List[str] = Field(
        ...,
        description="List of possible answers",
        examples=[["Paris"]]
    )
    document_url: Optional[str] = Field(
        default=None,
        description="URL to the source document",
        examples=["https://en.wikipedia.org/wiki/France"]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the question"
    )


class NQBulkUploadRequest(BaseModel):
    """Request model for bulk uploading NQ data."""
    
    collection: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Collection name to store NQ documents",
        examples=["natural_questions"]
    )
    questions: List[NQQuestionItem] = Field(
        ...,
        description="List of NQ question items to upload",
        min_length=1,
        max_length=100
    )
    chunk_size: Optional[int] = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Chunk size in characters",
        examples=[1000, 1500, 2000]
    )
    chunk_overlap: Optional[int] = Field(
        default=200,
        ge=0,
        le=500,
        description="Chunk overlap in characters",
        examples=[100, 200]
    )
    process_immediately: Optional[bool] = Field(
        default=True,
        description="Whether to process documents immediately (parse, chunk, embed)",
        examples=[True, False]
    )


class NQBulkUploadResponse(BaseModel):
    """Response model for bulk NQ upload operation."""
    
    success: bool = Field(description="Whether the upload was successful")
    uploaded_count: int = Field(description="Number of questions uploaded")
    collection: str = Field(description="Collection name")
    message: str = Field(description="Status message")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class NQUploadTaskResponse(BaseModel):
    """Response model for NQ upload task with task ID."""
    
    task_id: str = Field(
        description="Task ID for tracking the upload process",
        examples=["abc123-def456-ghi789"]
    )
    collection: str = Field(description="Collection name")
    question_count: int = Field(description="Number of questions in the batch")
    status: str = Field(
        description="Task status",
        examples=["PENDING", "STARTED", "SUCCESS", "FAILURE"]
    )
    message: str = Field(
        description="Status message",
        examples=["NQ bulk upload queued for processing"]
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "collection": "natural_questions",
                "question_count": 100,
                "status": "PENDING",
                "message": "NQ bulk upload queued for processing",
                "created_at": "2024-01-01T12:00:00"
            }
        }
    }


class NQCrawlRequest(BaseModel):
    """Request model for triggering NQ dataset crawl."""
    
    collection: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Collection name to store crawled NQ documents",
        examples=["natural_questions"]
    )
    dataset_url: Optional[str] = Field(
        default="https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl",
        description="URL to NQ dataset JSONL file",
        examples=["https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"]
    )
    batch_size: Optional[int] = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of questions per batch",
        examples=[50, 100, 200]
    )
    max_questions: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of questions to crawl (None = all)",
        examples=[100, 1000, None]
    )
    chunk_size: Optional[int] = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Chunk size in characters"
    )
    chunk_overlap: Optional[int] = Field(
        default=200,
        ge=0,
        le=500,
        description="Chunk overlap in characters"
    )


class NQCrawlTaskResponse(BaseModel):
    """Response model for NQ crawl task."""
    
    task_id: str = Field(
        description="Task ID for tracking the crawl process",
        examples=["xyz789-abc123-def456"]
    )
    collection: str = Field(description="Collection name")
    dataset_url: str = Field(description="URL of the dataset being crawled")
    status: str = Field(
        description="Task status",
        examples=["PENDING", "STARTED", "SUCCESS", "FAILURE"]
    )
    message: str = Field(
        description="Status message",
        examples=["NQ dataset crawl started"]
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Crawl start timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "xyz789-abc123-def456",
                "collection": "natural_questions",
                "dataset_url": "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl",
                "status": "PENDING",
                "message": "NQ dataset crawl started",
                "created_at": "2024-01-01T12:00:00"
            }
        }
    }
