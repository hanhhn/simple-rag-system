"""
Query and RAG endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException

from src.core.logging import get_logger
from src.core.exceptions import ServiceError, ValidationError
from src.api.models.query import QueryRequest, QueryResponse
from src.api.dependencies import get_query_processor
from src.utils.validators import QueryValidator


logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    query_processor = Depends(get_query_processor)
) -> QueryResponse:
    """
    Process a query through the RAG pipeline.
    
    This endpoint:
    1. Embeds the query
    2. Searches for relevant documents
    3. Generates an answer using the LLM with context
    
    Args:
        request: Query request with query text, collection, and options
        
    Returns:
        QueryResponse with answer and retrieved documents
    """
    validator = QueryValidator()
    
    try:
        # Validate query
        validator.validate_query(request.query)
        validator.validate_top_k(request.top_k)
        
        logger.info(
            "Processing query request",
            query=request.query[:100],
            collection=request.collection,
            top_k=request.top_k,
            use_rag=request.use_rag
        )
        
        # Process query
        if request.stream:
            # Streaming response
            result = query_processor.process_query_stream(
                query=request.query,
                collection_name=request.collection,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
        else:
            # Regular response
            result = query_processor.process_query(
                query=request.query,
                collection_name=request.collection,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                use_rag=request.use_rag
            )
        
        # Format response
        response = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            answer_chunks=result.get("answer_chunks"),
            retrieved_documents=[
                {
                    "id": doc["id"],
                    "score": doc["score"],
                    "text": doc.get("payload", {}).get("text", ""),
                    "metadata": doc.get("payload", {})
                }
                for doc in result["retrieved_documents"]
            ],
            retrieval_count=result["retrieval_count"],
            collection=result["collection"],
            top_k=result["top_k"],
            score_threshold=result["score_threshold"],
            use_rag=request.use_rag
        )
        
        logger.info(
            "Query processed successfully",
            query=request.query[:100],
            has_answer=bool(result["answer"]),
            retrieval_count=result["retrieval_count"]
        )
        
        return response
        
    except ValidationError as e:
        logger.error("Query validation failed", error=str(e))
        raise HTTPException(
            status_code=400,
            detail=e.to_dict()
        )
    except ServiceError as e:
        logger.error("Query service error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error("Unexpected error during query processing", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": f"Failed to process query: {str(e)}"
            }
        )


@router.post("/stream", response_model=QueryResponse)
async def process_query_stream(
    request: QueryRequest,
    query_processor = Depends(get_query_processor)
) -> QueryResponse:
    """
    Process a query with streaming response.
    
    Args:
        request: Query request
        
    Returns:
        QueryResponse with streaming answer chunks
    """
    # Set stream flag
    request.stream = True
    
    # Use regular query processor
    return await process_query(request, query_processor)
