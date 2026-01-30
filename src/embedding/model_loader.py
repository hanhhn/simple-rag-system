"""
Model loader for embedding models.
"""
from pathlib import Path
from typing import Optional

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingModelNotFoundError
from src.core.config import get_config


logger = get_logger(__name__)


class ModelLoader:
    """
    Loader for embedding models.
    
    This class handles loading and managing embedding models,
    including caching models in memory and handling downloads.
    
    Attributes:
        cache_dir: Directory to cache downloaded models
        device: Device to load models on (cpu/cuda)
        
    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load("sentence-transformers/all-MiniLM-L6-v2")
    """
    
    def __init__(self, cache_dir: Optional[Path | str] = None, device: Optional[str] = None) -> None:
        """
        Initialize model loader.
        
        Args:
            cache_dir: Directory to cache models (uses config default if None)
            device: Device to load models on (uses config default if None)
        """
        config = get_config()
        
        self.cache_dir = Path(cache_dir or config.storage.model_cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or config.embedding.device
        self._loaded_models: dict = {}
    
    def load_model(self, model_name: str) -> object:
        """
        Load an embedding model from Hugging Face or local path.
        
        Args:
            model_name: Name of the model on Hugging Face or local path
            
        Returns:
            Loaded sentence-transformers model
            
        Raises:
            EmbeddingModelNotFoundError: If model cannot be loaded
            
        Example:
            >>> loader = ModelLoader()
            >>> model = loader.load_model("sentence-transformers/all-MiniLM-L6-v2")
        """
        # Check if model is already loaded
        if model_name in self._loaded_models:
            logger.info("Model already loaded in memory", model=model_name)
            return self._loaded_models[model_name]
        
        logger.info("Loading embedding model", model=model_name, device=self.device)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model with caching
            model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            
            # Cache in memory
            self._loaded_models[model_name] = model
            
            logger.info(
                "Model loaded successfully",
                model=model_name,
                device=self.device,
                dimension=model.get_sentence_embedding_dimension()
            )
            
            return model
            
        except Exception as e:
            logger.error("Failed to load model", model=model_name, error=str(e))
            raise EmbeddingModelNotFoundError(
                f"Failed to load embedding model '{model_name}': {str(e)}",
                details={"model_name": model_name, "error": str(e)}
            )
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory cache.
        
        Args:
            model_name: Name of the model to unload
            
        Example:
            >>> loader = ModelLoader()
            >>> loader.unload_model("sentence-transformers/all-MiniLM-L6-v2")
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info("Model unloaded from memory", model=model_name)
    
    def unload_all(self) -> None:
        """
        Unload all models from memory cache.
        
        Example:
            >>> loader = ModelLoader()
            >>> loader.unload_all()
        """
        count = len(self._loaded_models)
        self._loaded_models.clear()
        logger.info("All models unloaded from memory", count=count)
    
    def get_model_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Embedding dimension
            
        Raises:
            EmbeddingModelNotFoundError: If model cannot be loaded
            
        Example:
            >>> loader = ModelLoader()
            >>> dim = loader.get_model_dimension("sentence-transformers/all-MiniLM-L6-v2")
            >>> print(dim)  # 384
        """
        model = self.load_model(model_name)
        return model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """
        Clear the model cache directory.
        
        Warning: This will delete downloaded models, requiring
        them to be downloaded again.
        
        Example:
            >>> loader = ModelLoader()
            >>> loader.clear_cache()
        """
        import shutil
        
        logger.warning("Clearing model cache", cache_dir=str(self.cache_dir))
        
        # Unload all models first
        self.unload_all()
        
        # Clear cache directory
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                logger.info("Removed model from cache", model=item.name)
    
    def get_cache_size(self) -> int:
        """
        Get the size of the model cache directory in bytes.
        
        Returns:
            Cache directory size in bytes
            
        Example:
            >>> loader = ModelLoader()
            >>> size_mb = loader.get_cache_size() / (1024 * 1024)
            >>> print(f"Cache size: {size_mb:.1f} MB")
        """
        total_size = 0
        
        if self.cache_dir.exists():
            for item in self.cache_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        
        return total_size
    
    def list_cached_models(self) -> list[str]:
        """
        List all models in the cache directory.
        
        Returns:
            List of model names/directories
            
        Example:
            >>> loader = ModelLoader()
            >>> models = loader.list_cached_models()
            >>> for model in models:
            ...     print(model)
        """
        models = []
        
        if self.cache_dir.exists():
            # Look for model directories
            for item in self.cache_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    models.append(item.name)
        
        return sorted(models)
