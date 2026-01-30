"""
Natural Questions (NQ) crawler tasks for Celery.

This module contains Celery tasks for crawling, downloading, and processing
the Natural Questions dataset from Google Research.
"""
import json
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
from io import BytesIO
from tqdm import tqdm

from src.tasks.celery_app import celery_app
from src.core.logging import get_logger
from src.core.exceptions import ServiceError
from src.services.document_processor import DocumentProcessor
from src.services.storage_manager import StorageManager


logger = get_logger(__name__)


def download_nq_dataset(dataset_url: str, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Download and parse NQ dataset from the given URL.
    
    Args:
        dataset_url: URL to the NQ dataset JSONL file
        max_questions: Maximum number of questions to parse (None = all)
        
    Returns:
        List of parsed question dictionaries
        
    Raises:
        ServiceError: If download or parsing fails
    """
    try:
        logger.info("Starting NQ dataset download", url=dataset_url)
        
        # Download the dataset
        response = requests.get(dataset_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Parse JSONL lines
        questions = []
        total_lines = 0
        
        # Get total file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        # Use tqdm for progress tracking
        with tqdm(
            desc="Downloading NQ dataset",
            unit='B',
            unit_scale=True,
            total=total_size
        ) as pbar:
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        
                        # Extract question and answers
                        # NQ-open format: question and answer fields
                        question_text = data.get("question", "")
                        answers = data.get("answer", [])
                        document_url = data.get("document", {}).get("url", "")
                        
                        # Skip if no question or answer
                        if not question_text or not answers:
                            continue
                        
                        # Format as NQ document
                        question_data = {
                            "question": question_text,
                            "answer": answers if isinstance(answers, list) else [answers],
                            "document_url": document_url,
                            "metadata": {
                                "source": "natural-questions",
                                "dataset_url": dataset_url,
                                "raw_data": data
                            }
                        }
                        
                        questions.append(question_data)
                        total_lines += 1
                        
                        # Check max questions limit
                        if max_questions and len(questions) >= max_questions:
                            logger.info(
                                "Reached max questions limit",
                                count=len(questions),
                                limit=max_questions
                            )
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON line", error=str(e))
                        continue
                    except Exception as e:
                        logger.warning("Error processing question line", error=str(e))
                        continue
                    
                    # Update progress bar
                    pbar.update(len(line))
        
        logger.info(
            "NQ dataset download completed",
            total_questions=len(questions),
            total_lines_processed=total_lines
        )
        
        return questions
        
    except requests.RequestException as e:
        logger.error("Failed to download NQ dataset", error=str(e), url=dataset_url)
        raise ServiceError(
            error_code="DOWNLOAD_FAILED",
            message=f"Failed to download NQ dataset: {str(e)}"
        )
    except Exception as e:
        logger.error("Unexpected error during NQ dataset download", error=str(e))
        raise ServiceError(
            error_code="INTERNAL_ERROR",
            message=f"Unexpected error during download: {str(e)}"
        )


def format_nq_documents(
    questions: List[Dict[str, Any]],
    collection: str
) -> List[str]:
    """
    Format NQ questions into document text format.
    
    Args:
        questions: List of question dictionaries
        collection: Collection name
        
    Returns:
        List of formatted document texts
    """
    documents = []
    
    for q in questions:
        # Format as Q&A document
        doc_text = f"Question: {q['question']}\n\n"
        doc_text += f"Answer(s): {', '.join(q['answer'])}\n"
        
        if q.get('document_url'):
            doc_text += f"\nSource: {q['document_url']}"
        
        documents.append(doc_text)
    
    return documents


def save_nq_documents(
    documents: List[str],
    collection: str,
    storage_manager: StorageManager
) -> List[str]:
    """
    Save NQ documents to storage.
    
    Args:
        documents: List of document texts
        collection: Collection name
        storage_manager: Storage manager instance
        
    Returns:
        List of storage paths
    """
    storage_paths = []
    
    for i, doc_text in enumerate(documents):
        # Generate filename
        filename = f"nq_doc_{i:06d}.txt"
        
        # Save to storage
        content = doc_text.encode('utf-8')
        storage_path = storage_manager.save_file(content, filename, collection)
        storage_paths.append(str(storage_path))
    
    return storage_paths


@celery_app.task(
    name="src.tasks.nq_crawler_task.process_nq_bulk_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def process_nq_bulk_task(
    self,
    collection: str,
    questions: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    process_immediately: bool = True
) -> Dict[str, Any]:
    """
    Process a bulk batch of Natural Questions asynchronously.
    
    This task formats NQ questions into documents and optionally
    processes them (chunk, embed, store).
    
    Args:
        collection: Collection name
        questions: List of question dictionaries
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        process_immediately: Whether to process immediately
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(
            "Starting NQ bulk processing task",
            collection=collection,
            question_count=len(questions),
            task_id=self.request.id
        )
        
        # Format questions into documents
        documents = format_nq_documents(questions, collection)
        
        # Save documents to storage
        storage_manager = StorageManager()
        storage_paths = save_nq_documents(documents, collection, storage_manager)
        
        results = {
            "success": True,
            "task_id": self.request.id,
            "collection": collection,
            "question_count": len(questions),
            "documents_created": len(documents),
            "storage_paths": storage_paths
        }
        
        # Process documents if requested
        if process_immediately:
            # Import here to avoid circular dependency
            from src.tasks import document_tasks
            from src.utils.helpers import generate_id
            
            # Process each document
            for i, (storage_path, question_data) in enumerate(zip(storage_paths, questions)):
                document_id = generate_id()
                filename = Path(storage_path).name
                
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "collection": collection,
                    "question": question_data.get("question"),
                    "answers": question_data.get("answer"),
                    "source": "natural-questions"
                }
                
                # Queue document processing task
                doc_task = document_tasks.process_document_task.delay(
                    storage_path=storage_path,
                    collection=collection,
                    filename=filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunker_type="sentence",
                    metadata=metadata
                )
                
                logger.info(
                    "Document processing queued",
                    filename=filename,
                    document_id=document_id,
                    doc_task_id=doc_task.id
                )
        
        logger.info(
            "NQ bulk processing completed",
            collection=collection,
            question_count=len(questions),
            task_id=self.request.id
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "Error in NQ bulk processing task",
            collection=collection,
            question_count=len(questions),
            error=str(e),
            task_id=self.request.id,
            exc_info=True
        )
        raise self.retry(exc=e)


@celery_app.task(
    name="src.tasks.nq_crawler_task.crawl_nq_task",
    bind=True,
    max_retries=3,
    default_retry_delay=120
)
def crawl_nq_task(
    self,
    collection: str,
    dataset_url: str = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl",
    batch_size: int = 100,
    max_questions: Optional[int] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Crawl Natural Questions dataset asynchronously.
    
    This task downloads, parses, and processes the NQ dataset in batches.
    
    Args:
        collection: Collection name to store documents
        dataset_url: URL to NQ dataset JSONL file
        batch_size: Number of questions per batch
        max_questions: Maximum questions to crawl (None = all)
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        
    Returns:
        Dictionary with crawl results
    """
    try:
        logger.info(
            "Starting NQ dataset crawl task",
            collection=collection,
            dataset_url=dataset_url,
            batch_size=batch_size,
            max_questions=max_questions,
            task_id=self.request.id
        )
        
        # Download and parse dataset
        all_questions = download_nq_dataset(dataset_url, max_questions)
        
        if not all_questions:
            raise ServiceError(
                error_code="NO_DATA",
                message="No questions found in the dataset"
            )
        
        # Process in batches
        total_batches = (len(all_questions) + batch_size - 1) // batch_size
        processed_count = 0
        task_ids = []
        
        for batch_idx in tqdm(
            range(total_batches),
            desc="Processing batches"
        ):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_questions))
            batch_questions = all_questions[start_idx:end_idx]
            
            # Queue batch processing task
            batch_task = process_nq_bulk_task.delay(
                collection=collection,
                questions=batch_questions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                process_immediately=True
            )
            
            task_ids.append(batch_task.id)
            processed_count += len(batch_questions)
            
            logger.info(
                "Batch processed",
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_size=len(batch_questions),
                batch_task_id=batch_task.id
            )
        
        logger.info(
            "NQ dataset crawl completed",
            collection=collection,
            total_questions=len(all_questions),
            total_batches=total_batches,
            processed_count=processed_count,
            task_id=self.request.id
        )
        
        return {
            "success": True,
            "task_id": self.request.id,
            "collection": collection,
            "dataset_url": dataset_url,
            "total_questions": len(all_questions),
            "total_batches": total_batches,
            "batch_size": batch_size,
            "processed_count": processed_count,
            "batch_task_ids": task_ids
        }
        
    except ServiceError as e:
        logger.error(
            "NQ dataset crawl failed",
            collection=collection,
            error=str(e),
            task_id=self.request.id
        )
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in NQ dataset crawl",
            collection=collection,
            error=str(e),
            task_id=self.request.id,
            exc_info=True
        )
        raise self.retry(exc=e)
