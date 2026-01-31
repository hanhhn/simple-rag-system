"""
Generic Sentence Transformer model implementation.
This provides a reusable implementation for any sentence-transformers model.
"""
import time
from typing import List, Optional

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingGenerationError
from src.core.config import get_config
from src.embedding.base import EmbeddingModel
from src.embedding.model_manager import ModelManager


logger = get_logger(__name__)


class SentenceTransformerModel(EmbeddingModel):
    """
    Optimized generic implementation for sentence-transformers models.
    
    This class wraps any sentence-transformers model, providing a clean
    interface for generating embeddings. It uses the ModelManager singleton
    to ensure efficient model loading and caching across the application.
    
    Features:
    - Singleton model management (models loaded only once)
    - Automatic model sharing between instances
    - Reference counting for proper cleanup
    - Thread-safe model loading
    
    Example:
        >>> model = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2", 384)
        >>> embedding = model.encode_single("Hello world")
    """
    
    def __init__(
        self,
        model_name: str,
        dimension: int,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        use_model_manager: bool = True
    ) -> None:
        """
        Initialize Sentence Transformer model with optional singleton management.
        
        Args:
            model_name: Model name or path from Hugging Face
            dimension: Embedding dimension of the model
            cache_dir: Directory to cache models (used only if use_model_manager=False)
            device: Device to load model on (cpu/cuda) (used only if use_model_manager=False)
            max_length: Maximum input sequence length
            use_model_manager: Use ModelManager singleton for efficient caching
        """
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            max_length=max_length
        )
        
        self._use_model_manager = use_model_manager
        self._model_manager: Optional[ModelManager] = None
        
        if use_model_manager:
            # Use ModelManager singleton for efficient caching
            self._model_manager = ModelManager.get_instance(cache_dir=cache_dir, device=device)
            self.model = self._model_manager.get_model(model_name)
            self._loader = None
        else:
            # Direct loading without ModelManager (for testing/edge cases)
            from src.embedding.model_loader import ModelLoader
            self._loader = ModelLoader(cache_dir=cache_dir, device=device)
            self.model = self._loader.load_model(model_name)
            self._model_manager = None
        
        logger.info(
            "Sentence Transformer model initialized",
            model=model_name,
            dimension=dimension,
            max_length=max_length,
            use_model_manager=use_model_manager
        )
    
    def __del__(self) -> None:
        """
        Cleanup when model instance is destroyed.
        
        Releases reference to model in ModelManager if used.
        """
        if self._use_model_manager and self._model_manager is not None:
            try:
                self._model_manager.release_model(self.model_name)
            except Exception as e:
                logger.warning("Failed to release model", model=self.model_name, error=str(e))
    
    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode multiple texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingGenerationError: If encoding fails
            
        Example:
            >>> model = SentenceTransformerModel("model-name", 384)
            >>> embeddings = model.encode(["text1", "text2"], batch_size=16)
            >>> print(len(embeddings))  # 2
        """
        if not texts:
            logger.debug("No texts provided for encoding", model=self.model_name)
            return []
        
        start_time = time.time()
        
        try:
            total_chars = sum(len(text) for text in texts)
            avg_length = total_chars / len(texts) if texts else 0
            
            logger.info(
                "Starting encoding batch",
                count=len(texts),
                batch_size=batch_size,
                model=self.model_name,
                dimension=self.dimension,
                total_chars=total_chars,
                avg_text_length=f"{avg_length:.1f}"
            )
            
            # Encode texts with normalized embeddings
            encode_start = time.time()
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            encode_elapsed = time.time() - encode_start
            
            # Convert to list format
            convert_start = time.time()
            result = [embedding.tolist() for embedding in embeddings]
            convert_elapsed = time.time() - convert_start
            
            total_elapsed = time.time() - start_time
            
            logger.info(
                "Successfully encoded texts",
                count=len(texts),
                dimension=self.dimension,
                model=self.model_name,
                encode_time=f"{encode_elapsed:.4f}s",
                convert_time=f"{convert_elapsed:.4f}s",
                total_time=f"{total_elapsed:.4f}s",
                avg_time_per_text=f"{total_elapsed / len(texts) * 1000:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Failed to encode texts",
                error=str(e),
                error_type=type(e).__name__,
                count=len(texts),
                model=self.model_name,
                elapsed_time=f"{elapsed:.4f}s"
            )
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings with model '{self.model_name}': {str(e)}",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "text_count": len(texts),
                    "model_name": self.model_name,
                    "elapsed_time": f"{elapsed:.4f}s"
                }
            )
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingGenerationError: If encoding fails
            
        Example:
            >>> model = SentenceTransformerModel("model-name", 384)
            >>> embedding = model.encode_single("Hello world")
            >>> print(len(embedding))  # 384
        """
        try:
            # Use batch encoding for single text (more efficient)
            embeddings = self.encode([text])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error(
                "Failed to encode text",
                text=text[:100],
                error=str(e),
                model=self.model_name
            )
            raise EmbeddingGenerationError(
                f"Failed to generate embedding with model '{self.model_name}': {str(e)}",
                details={
                    "error": str(e),
                    "text_preview": text[:100],
                    "model_name": self.model_name
                }
            )
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
            
        Example:
            >>> model = SentenceTransformerModel("model-name", 384)
            >>> emb1 = model.encode_single("Hello")
            >>> emb2 = model.encode_single("Hi")
            >>> similarity = model.compute_similarity(emb1, emb2)
            >>> print(similarity)  # e.g., 0.85
        """
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))
