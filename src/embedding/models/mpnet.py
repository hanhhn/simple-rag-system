"""
MPNet embedding model implementation.
"""
from typing import List

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingGenerationError
from src.embedding.base import EmbeddingModel
from src.embedding.model_loader import ModelLoader


logger = get_logger(__name__)


class MPNetModel(EmbeddingModel):
    """
    MPNet embedding model implementation.
    
    This class wraps the sentence-transformers MPNet model,
    providing a clean interface for generating embeddings.
    MPNet generally provides better quality embeddings than MiniLM
    but is larger and slower.
    
    Model: sentence-transformers/all-mpnet-base-v2
    Dimension: 768
    Max Length: 512
    
    Example:
        >>> model = MPNetModel()
        >>> embedding = model.encode_single("Hello world")
        >>> print(len(embedding))  # 768
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    DIMENSION = 768
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        device: str | None = None,
        lazy_load: bool = True
    ) -> None:
        """
        Initialize MPNet model.
        
        Args:
            model_name: Model name or path (default: all-mpnet-base-v2)
            cache_dir: Directory to cache models
            device: Device to load model on (cpu/cuda)
            lazy_load: If True, load model on first use (safer for multiprocessing)
        """
        super().__init__(model_name=model_name, dimension=self.DIMENSION)

        self.loader = ModelLoader(cache_dir=cache_dir, device=device)
        self._model_name = model_name

        if lazy_load:
            # Lazy loading - safer for multiprocessing (Celery workers)
            self.model = None
            logger.info(
                "MPNet model initialized (lazy loading)",
                model=model_name,
                dimension=self.DIMENSION
            )
        else:
            # Eager loading
            self.model = self.loader.load_model(model_name)
            logger.info("MPNet model initialized", model=model_name, dimension=self.DIMENSION)

    @property
    def _loaded_model(self):
        """Lazy load of model when first accessed."""
        logger.debug("Accessing _loaded_model property", model=self._model_name, has_model=self.model is not None)

        if self.model is None:
            logger.info("Loading model on first access", model=self._model_name)
            try:
                self.model = self.loader.load_model(self._model_name)
                logger.info("Model loaded successfully", model=self._model_name)
            except Exception:
                logger.error(
                    "Failed to load model in _loaded_model property",
                    model=self._model_name,
                    exc_info=True
                )
                raise

        logger.debug("Returning model from _loaded_model property", model=self._model_name)
        return self.model

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
            >>> model = MPNetModel()
            >>> embeddings = model.encode(["text1", "text2"], batch_size=16)
            >>> print(len(embeddings))  # 2
        """
        if not texts:
            return []

        logger.info(
            "Starting encode operation",
            count=len(texts),
            batch_size=batch_size
        )

        try:
            logger.debug(
                "Encoding texts",
                count=len(texts),
                batch_size=batch_size
            )

            # Encode texts using lazy-loaded model
            model = self._loaded_model
            logger.debug("Model loaded, starting encoding")

            try:
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                logger.debug("Encoding completed, converting to list")

                # Convert to list format
                result = [embedding.tolist() for embedding in embeddings]

                logger.info(
                    "Successfully encoded texts",
                    count=len(texts),
                    dimension=self.DIMENSION
                )

                return result

            except Exception as encode_error:
                logger.error(
                    "Model.encode() failed",
                    error=str(encode_error),
                    error_type=type(encode_error).__name__,
                    text_count=len(texts),
                    batch_size=batch_size,
                    exc_info=True
                )
                raise EmbeddingGenerationError(
                    f"Failed to generate embeddings: {str(encode_error)}",
                    details={
                        "error": str(encode_error),
                        "error_type": type(encode_error).__name__,
                        "text_count": len(texts)
                    }
                )
        except Exception as e:
            logger.error(
                "Failed to encode texts",
                error=str(e),
                error_type=type(e).__name__,
                count=len(texts)
            )
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings: {str(e)}",
                details={"error": str(e), "text_count": len(texts)}
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
            >>> model = MPNetModel()
            >>> embedding = model.encode_single("Hello world")
            >>> print(len(embedding))  # 768
        """
        try:
            # Use batch encoding for single text (more efficient)
            embeddings = self.encode([text])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error("Failed to encode text", text=text[:100], error=str(e))
            raise EmbeddingGenerationError(
                f"Failed to generate embedding: {str(e)}",
                details={"error": str(e), "text_preview": text[:100]}
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
            >>> model = MPNetModel()
            >>> emb1 = model.encode_single("Hello")
            >>> emb2 = model.encode_single("Hi")
            >>> similarity = model.compute_similarity(emb1, emb2)
            >>> print(similarity)  # e.g., 0.90
        """
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
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
