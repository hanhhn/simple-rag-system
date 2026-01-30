"""
Embedding service for managing embedding generation.
"""
from typing import List, Optional

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingError
from src.core.config import get_config
from src.embedding.base import EmbeddingModel
from src.embedding.models.minilm import MiniLMModel
from src.embedding.models.mpnet import MPNetModel
from src.embedding.cache import EmbeddingCache


logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating and managing embeddings.
    
    This class provides a high-level interface for embedding generation,
    including caching, batch processing, and model management.
    
    Example:
        >>> service = EmbeddingService()
        >>> embeddings = service.generate_embeddings(["text1", "text2"])
        >>> print(len(embeddings))  # 2
    """
    
    MODEL_REGISTRY = {
        "minilm": MiniLMModel,
        "mpnet": MPNetModel,
        "sentence-transformers/all-MiniLM-L6-v2": MiniLMModel,
        "sentence-transformers/all-mpnet-base-v2": MPNetModel,
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> None:
        """
        Initialize embedding service.
        
        Args:
            model_name: Model name (uses config default if None)
            use_cache: Whether to use embedding cache
        """
        config = get_config()
        
        self.model_name = model_name or config.embedding.model_name
        self.use_cache = use_cache
        
        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=config.storage.cache_path,
            enabled=use_cache
        )
        
        # Lazy load model - don't load until first use
        # This avoids issues with loading models in forked processes
        self._model: Optional[EmbeddingModel] = None
        
        logger.info(
            "Embedding service initialized (lazy loading)",
            model=self.model_name,
            use_cache=use_cache
        )
    
    @property
    def model(self) -> EmbeddingModel:
        """
        Get the embedding model, loading it lazily if needed.
        
        This property ensures the model is only loaded when first accessed,
        which is safer for multiprocessing contexts (e.g., Celery workers).
        
        Returns:
            Initialized embedding model
        """
        if self._model is None:
            self._model = self._load_model(self.model_name)
            logger.info(
                "Model loaded (lazy initialization)",
                model=self.model_name,
                dimension=self._model.get_dimension()
            )
        return self._model
    
    def _load_model(self, model_name: str) -> EmbeddingModel:
        """
        Load the embedding model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Initialized embedding model
        """
        # Check if model is in registry
        for key, model_class in self.MODEL_REGISTRY.items():
            if key in model_name.lower():
                logger.info("Loading model from registry", model=model_name)
                return model_class()
        
        # If not found, try loading directly
        logger.info("Loading model directly", model=model_name)
        return MiniLMModel(model_name=model_name)
    
    def generate_embedding(self, text: str, use_cache_override: Optional[bool] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache_override: Override cache setting for this call
            
        Returns:
            Embedding vector
            
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.generate_embedding("Hello world")
            >>> print(len(embedding))  # model dimension
        """
        use_cache = use_cache_override if use_cache_override is not None else self.use_cache
        
        # Check cache
        if use_cache:
            cached_embedding = self.cache.get(text)
            if cached_embedding:
                logger.debug("Embedding cache hit", text_length=len(text))
                return cached_embedding
        
        # Generate embedding
        try:
            embedding = self.model.encode_single(text)
            
            # Cache result
            if use_cache:
                self.cache.set(text, embedding)
            
            logger.debug(
                "Embedding generated",
                text_length=len(text),
                dimension=len(embedding),
                cached=False
            )
            
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}",
                details={"text_preview": text[:100], "error": str(e)}
            )
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache_override: Optional[bool] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache_override: Override cache setting for this call
            
        Returns:
            List of embedding vectors
            
        Example:
            >>> service = EmbeddingService()
            >>> embeddings = service.generate_embeddings(["text1", "text2"])
            >>> print(len(embeddings))  # 2
        """
        if not texts:
            return []
        
        use_cache = use_cache_override if use_cache_override is not None else self.use_cache
        
        # Check cache for all texts
        if use_cache:
            cached_embeddings = self.cache.get_batch(texts)
            
            # Generate embeddings for cache misses
            texts_to_encode = [
                text for text, cached in zip(texts, cached_embeddings)
                if cached is None
            ]
            
            if texts_to_encode:
                # Generate embeddings for cache misses
                new_embeddings = self.model.encode(texts_to_encode, batch_size=batch_size)
                
                # Cache new embeddings
                if use_cache:
                    self.cache.set_batch(texts_to_encode, new_embeddings)
                
                # Combine cached and new embeddings
                embeddings = []
                cache_idx = 0
                for text in texts:
                    if cached_embeddings[cache_idx] is not None:
                        embeddings.append(cached_embeddings[cache_idx])
                    else:
                        embeddings.append(new_embeddings[cache_idx])
                    cache_idx += 1
                
                logger.info(
                    "Embeddings generated with cache",
                    total=len(texts),
                    cached=len(texts) - len(texts_to_encode),
                    generated=len(new_embeddings)
                )
                
                return embeddings
            else:
                # All cached
                logger.info("All embeddings from cache", count=len(texts))
                return cached_embeddings
        else:
            # No caching, generate all
            try:
                embeddings = self.model.encode(texts, batch_size=batch_size)
                
                logger.info(
                    "Embeddings generated without cache",
                    count=len(texts),
                    batch_size=batch_size
                )
                
                return embeddings
                
            except Exception as e:
                logger.error("Failed to generate embeddings", count=len(texts), error=str(e))
                raise EmbeddingError(
                    f"Failed to generate embeddings: {str(e)}",
                    details={"text_count": len(texts), "error": str(e)}
                )
    
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_dimension()
    
    def get_model_name(self) -> str:
        """
        Get the current model name.
        
        Returns:
            Model name
        """
        return self.model_name
    
    def save_cache(self) -> None:
        """
        Save the embedding cache to disk.
        
        Example:
            >>> service = EmbeddingService()
            >>> service.save_cache()
        """
        self.cache.save()
        logger.info("Embedding cache saved")
    
    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        
        Example:
            >>> service = EmbeddingService()
            >>> service.clear_cache()
        """
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return self.cache.get_stats()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self.model.compute_similarity(embedding1, embedding2)
