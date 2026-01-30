"""
Natural Questions (NQ) document management endpoints.
"""
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from src.core.logging import get_logger
from src.core.exceptions import ServiceError
from src.utils.helpers import generate_id
from src.api.models.nq_document import (
    NQBulkUploadRequest,
    NQBulkUploadResponse,
    NQUploadTaskResponse,
    NQCrawlRequest,
    NQCrawlTaskResponse
)
from src.tasks.nq_crawler_task import (
    process_nq_bulk_task,
    crawl_nq_task
)


logger = get_logger(__name__)

router = APIRouter(prefix="/nq", tags=["Natural Questions"])


@router.post(
    "/bulk-upload",
    response_model=NQUploadTaskResponse,
    status_code=202,
    summary="Bulk upload Natural Questions",
    description="""
    Upload a batch of Natural Questions for asynchronous processing.
    
    This endpoint accepts a batch of questions and answers from the Natural Questions dataset,
    and queues them for background processing (format, chunk, embed, store).
    
    **Parameters:**
    - `collection`: Collection name to store documents in
    - `questions`: List of NQ question items (up to 100 per batch)
    - `chunk_size`: Chunk size in characters (default: 1000)
    - `chunk_overlap`: Chunk overlap in characters (default: 200)
    - `process_immediately`: Whether to process immediately (default: True)
    
    **Returns:**
    - Task response with task_id (use this to check processing status via Tasks API)
    """
)
async def bulk_upload_nq(
    request: NQBulkUploadRequest
) -> NQUploadTaskResponse:
    """
    Bulk upload Natural Questions for asynchronous processing.
    
    This endpoint accepts a batch of questions and answers, and queues
    them for background processing.
    
    Args:
        request: Bulk upload request with collection and questions
        
    Returns:
        NQUploadTaskResponse with task ID and status
    """
    try:
        logger.info(
            "NQ bulk upload requested",
            collection=request.collection,
            question_count=len(request.questions),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Prepare questions data for task
        questions_data = [
            {
                "question": item.question,
                "answer": item.answer,
                "document_url": item.document_url,
                "metadata": item.metadata
            }
            for item in request.questions
        ]
        
        # Generate task ID
        task_id = generate_id()
        
        # Queue bulk processing task
        task = process_nq_bulk_task.apply_async(
            kwargs={
                "collection": request.collection,
                "questions": questions_data,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "process_immediately": request.process_immediately
            }
        )
        
        logger.info(
            "NQ bulk upload task queued",
            collection=request.collection,
            question_count=len(request.questions),
            task_id=task.id
        )
        
        return NQUploadTaskResponse(
            task_id=task.id,
            collection=request.collection,
            question_count=len(request.questions),
            status="PENDING",
            message="NQ bulk upload queued for processing"
        )
        
    except ServiceError as e:
        logger.error("NQ bulk upload service error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error("Unexpected error during NQ bulk upload", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": f"Failed to upload NQ data: {str(e)}"
            }
        )


@router.post(
    "/crawl",
    response_model=NQCrawlTaskResponse,
    status_code=202,
    summary="Crawl Natural Questions dataset",
    description="""
    Trigger a crawl job to fetch Natural Questions dataset from GitHub repository.
    
    This endpoint initiates an asynchronous crawl task that:
    1. Downloads the NQ dataset JSONL file from GitHub
    2. Parses the dataset into structured questions and answers
    3. Pushes data to the bulk API endpoint in batches
    4. Processes documents (chunk, embed, store) if enabled
    
    **Parameters:**
    - `collection`: Collection name to store crawled documents
    - `dataset_url`: URL to NQ dataset JSONL file (default: dev set)
    - `batch_size`: Number of questions per batch (default: 100)
    - `max_questions`: Maximum questions to crawl (default: all)
    - `chunk_size`: Chunk size in characters (default: 1000)
    - `chunk_overlap`: Chunk overlap in characters (default: 200)
    
    **Returns:**
    - Task response with task_id (use this to check crawl status via Tasks API)
    """
)
async def crawl_nq_dataset(
    request: NQCrawlRequest
) -> NQCrawlTaskResponse:
    """
    Trigger Natural Questions dataset crawl.
    
    This endpoint initiates a background crawl task that downloads
    and processes the NQ dataset.
    
    Args:
        request: Crawl request with dataset URL and parameters
        
    Returns:
        NQCrawlTaskResponse with task ID and status
    """
    try:
        logger.info(
            "NQ dataset crawl requested",
            collection=request.collection,
            dataset_url=request.dataset_url,
            batch_size=request.batch_size,
            max_questions=request.max_questions
        )
        
        # Queue crawl task
        task = crawl_nq_task.apply_async(
            kwargs={
                "collection": request.collection,
                "dataset_url": request.dataset_url,
                "batch_size": request.batch_size,
                "max_questions": request.max_questions,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap
            }
        )
        
        logger.info(
            "NQ dataset crawl task queued",
            collection=request.collection,
            dataset_url=request.dataset_url,
            task_id=task.id
        )
        
        return NQCrawlTaskResponse(
            task_id=task.id,
            collection=request.collection,
            dataset_url=request.dataset_url,
            status="PENDING",
            message="NQ dataset crawl started"
        )
        
    except ServiceError as e:
        logger.error("NQ crawl service error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error("Unexpected error during NQ crawl", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": f"Failed to start NQ crawl: {str(e)}"
            }
        )
