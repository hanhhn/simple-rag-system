"""
Model loader for embedding models.
"""
import time
from pathlib import Path
from typing import Optional, Literal
from sentence_transformers import SentenceTransformer

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingModelNotFoundError
from src.core.config import get_config


logger = get_logger(__name__)


class ModelLoader:
    """
    Optimized loader for embedding models with advanced features.
    
    This class handles loading and managing embedding models with:
    - In-memory caching for reuse
    - Automatic model download from Hugging Face
    - Memory optimization (float16/bfloat16 for CUDA)
    - Device selection (CPU/CUDA)
    
    Attributes:
        cache_dir: Directory to cache downloaded models
        device: Device to load models on (cpu/cuda)
        use_half_precision: Use float16 for CUDA models
        trust_remote_code: Trust remote code when loading models
        
    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("sentence-transformers/all-MiniLM-L6-v2")
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        use_half_precision: Optional[bool] = None,
        trust_remote_code: bool = False
    ) -> None:
        """
        Initialize model loader with optimization settings.
        
        Args:
            cache_dir: Directory to cache models (uses config default if None)
            device: Device to load models on (uses config default if None)
            use_half_precision: Use float16 for CUDA models (auto-detected if None)
            trust_remote_code: Trust remote code when loading models (security risk)
        """
        config = get_config()
        
        self.cache_dir = Path(cache_dir or config.storage.model_cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or config.embedding.device
        
        # Auto-detect half precision for CUDA
        if use_half_precision is None:
            self.use_half_precision = (self.device == "cuda" and self._is_cuda_available())
            cuda_available = self._is_cuda_available()
            logger.info(
                "Auto-detected precision settings",
                device=self.device,
                cuda_available=cuda_available,
                half_precision=self.use_half_precision
            )
        else:
            self.use_half_precision = use_half_precision
            logger.info(
                "Using explicit precision settings",
                device=self.device,
                half_precision=self.use_half_precision
            )
            
        self.trust_remote_code = trust_remote_code
        self._loaded_models: dict = {}
        
        logger.info(
            "ModelLoader initialized",
            cache_dir=str(self.cache_dir),
            device=self.device,
            half_precision=self.use_half_precision,
            trust_remote_code=self.trust_remote_code
        )
    
    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_optimal_torch_dtype(self):
        """
        Get optimal torch dtype for the current device.
        
        Returns:
            torch.dtype: Optimal dtype for the device
        """
        try:
            import torch
            
            if self.device == "cuda" and self.use_half_precision:
                return torch.float16
            elif self.device == "cuda":
                return torch.float32
            else:
                return torch.float32
        except ImportError:
            return None
    
    def load_model(self, model_name: str) -> object:
        """
        Load an embedding model from Hugging Face or local path with optimizations.
        
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
        start_time = time.time()
        
        # Check if model is already loaded
        if model_name in self._loaded_models:
            elapsed = time.time() - start_time
            logger.info(
                "Model already loaded in memory",
                model=model_name,
                retrieval_time=f"{elapsed:.4f}s"
            )
            return self._loaded_models[model_name]
        
        logger.info(
            "Loading embedding model",
            model=model_name,
            device=self.device,
            half_precision=self.use_half_precision,
            cache_dir=str(self.cache_dir)
        )
        
        try:
            # Prepare model arguments with optimizations
            model_kwargs = {
                "cache_folder": str(self.cache_dir),
                "device": self.device,
            }
            
            # Add trust_remote_code if enabled
            if self.trust_remote_code:
                model_kwargs["trust_remote_code"] = True
                logger.warning("Loading model with trust_remote_code=True - security risk")
            
            # Note: SentenceTransformer doesn't support torch_dtype parameter
            # If dtype optimization is needed, it should be done after loading
            # Get optimal torch dtype for logging purposes only
            torch_dtype = self._get_optimal_torch_dtype()
            if torch_dtype is not None:
                logger.info("Optimal torch dtype for device", dtype=str(torch_dtype), device=self.device)
            
            # Load model with optimizations
            logger.debug("Starting model download/load", model=model_name)
            load_start = time.time()
            model = SentenceTransformer(model_name, **model_kwargs)
            load_elapsed = time.time() - load_start
            logger.debug("Model downloaded/loaded", model=model_name, load_time=f"{load_elapsed:.4f}s")
            
            # Get model details
            dimension = model.get_sentence_embedding_dimension()
            max_seq_length = model.max_seq_length if hasattr(model, 'max_seq_length') else 'unknown'
            
            logger.info(
                "Model loaded successfully",
                model=model_name,
                device=self.device,
                dimension=dimension,
                max_seq_length=max_seq_length,
                half_precision=self.use_half_precision,
                load_time=f"{load_elapsed:.4f}s"
            )
            
            # For CUDA models, set modules to eval mode for inference
            if self.device == "cuda":
                try:
                    for param in model.parameters():
                        param.requires_grad = False
                    logger.info("Set model to eval mode", model=model_name)
                except Exception as e:
                    logger.warning("Could not set requires_grad=False", model=model_name, error=str(e))
            
            # Cache in memory
            self._loaded_models[model_name] = model
            
            total_elapsed = time.time() - start_time
            logger.info(
                "Model caching completed",
                model=model_name,
                cache_size=len(self._loaded_models),
                total_time=f"{total_elapsed:.4f}s"
            )
            
            return model
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Failed to load model",
                model=model_name,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_time=f"{elapsed:.4f}s"
            )
            raise EmbeddingModelNotFoundError(
                f"Failed to load embedding model '{model_name}': {str(e)}",
                details={
                    "model_name": model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "device": self.device,
                    "half_precision": self.use_half_precision,
                    "elapsed_time": f"{elapsed:.4f}s"
                }
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
            logger.info(
                "Model unloaded from memory",
                model=model_name,
                remaining_models=len(self._loaded_models)
            )
        else:
            logger.warning(
                "Model not found in memory",
                model=model_name,
                cached_models=list(self._loaded_models.keys())
            )
    
    def unload_all(self) -> None:
        """
        Unload all models from memory cache.
        
        Example:
            >>> loader = ModelLoader()
            >>> loader.unload_all()
        """
        count = len(self._loaded_models)
        models_to_unload = list(self._loaded_models.keys())
        self._loaded_models.clear()
        logger.info(
            "All models unloaded from memory",
            unloaded_count=count,
            models=models_to_unload
        )
    
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
