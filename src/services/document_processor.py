"""
Document processing service.
"""
from pathlib import Path
from typing import List, Dict, Optional

from src.core.logging import get_logger
from src.core.exceptions import DocumentProcessingError
from src.core.config import get_config
from src.parsers.base import get_parser_factory
from src.utils.text_chunker import Chunk, chunk_text

# Import parsers to ensure they are registered
import src.parsers  # noqa: F401


logger = get_logger(__name__)


class DocumentProcessor:
    """
    Service for processing documents.
    
    This class handles document parsing, text extraction,
    chunking, and preparation for embedding.
    
    Example:
        >>> processor = DocumentProcessor()
        >>> chunks = processor.process_document("document.txt", "my_collection")
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunker_type: str = "sentence"
    ) -> None:
        """
        Initialize document processor.
        
        Args:
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
            chunker_type: Type of chunker to use
        """
        config = get_config()
        
        self.chunk_size = chunk_size or config.document.chunk_size
        self.chunk_overlap = chunk_overlap or config.document.chunk_overlap
        self.chunker_type = chunker_type
        
        self.parser_factory = get_parser_factory()
        
        logger.info(
            "Document processor initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunker_type=chunker_type
        )
    
    def parse_document(self, filepath: Path | str) -> str:
        """
        Parse a document and extract its text.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If parsing fails
            
        Example:
            >>> processor = DocumentProcessor()
            >>> text = processor.parse_document("document.pdf")
        """
        try:
            logger.info("Parsing document", filepath=str(filepath))
            
            # Get appropriate parser
            parser = self.parser_factory.get_parser(filepath)
            
            # Parse document
            text = parser.parse(filepath)
            
            if not text:
                raise DocumentProcessingError(
                    "Document contains no text content",
                    details={"filepath": str(filepath)}
                )
            
            logger.info(
                "Document parsed successfully",
                filepath=str(filepath),
                text_length=len(text)
            )
            
            return text
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("Failed to parse document", filepath=str(filepath), error=str(e))
            raise DocumentProcessingError(
                f"Failed to parse document: {str(e)}",
                details={"filepath": str(filepath), "error": str(e)}
            )
    
    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk document text into manageable pieces.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
            
        Example:
            >>> processor = DocumentProcessor()
            >>> chunks = processor.chunk_document(document_text)
        """
        try:
            logger.debug(
                "Chunking document",
                text_length=len(text),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Chunk text
            chunks = chunk_text(
                text,
                chunker_type=self.chunker_type,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Add metadata to chunks
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            logger.info(
                "Document chunked successfully",
                chunk_count=len(chunks),
                avg_chunk_size=sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Failed to chunk document", error=str(e))
            raise DocumentProcessingError(
                f"Failed to chunk document: {str(e)}",
                details={"error": str(e)}
            )
    
    def process_document(
        self,
        filepath: Path | str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Process a document end-to-end (parse and chunk).
        
        Args:
            filepath: Path to the document file
            metadata: Optional metadata to attach
            
        Returns:
            Dictionary with processing results
            
        Example:
            >>> processor = DocumentProcessor()
            >>> result = processor.process_document("document.pdf")
            >>> print(result["chunks"])  # List of chunks
        """
        try:
            # Parse document
            text = self.parse_document(filepath)
            
            # Chunk document
            chunks = self.chunk_document(text, metadata)
            
            # Prepare result
            result = {
                "filepath": str(filepath),
                "text": text,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "total_length": len(text),
                "metadata": metadata or {}
            }
            
            logger.info(
                "Document processed successfully",
                filepath=str(filepath),
                chunk_count=len(chunks)
            )
            
            return result
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("Failed to process document", filepath=str(filepath), error=str(e))
            raise DocumentProcessingError(
                f"Failed to process document: {str(e)}",
                details={"filepath": str(filepath), "error": str(e)}
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of file extensions
        """
        return self.parser_factory.get_supported_extensions()
