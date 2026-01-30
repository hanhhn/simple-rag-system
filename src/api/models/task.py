"""
Task-related API models.
"""
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskResponse(BaseModel):
    """Response model for task status."""
    
    task_id: str = Field(description="Celery task ID")
    status: TaskStatus = Field(description="Task status")
    task_name: str = Field(description="Task name")
    result: Optional[dict] = Field(default=None, description="Task result (if completed)")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    traceback: Optional[str] = Field(default=None, description="Error traceback (if failed)")
    created_at: Optional[datetime] = Field(default=None, description="Task creation time")
    started_at: Optional[datetime] = Field(default=None, description="Task start time")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion time")
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Task progress (0.0-1.0)")
    metadata: dict = Field(default_factory=dict, description="Additional task metadata")


class TaskListResponse(BaseModel):
    """Response model for task list."""
    
    tasks: list[TaskResponse] = Field(description="List of tasks")
    total: int = Field(description="Total number of tasks")
