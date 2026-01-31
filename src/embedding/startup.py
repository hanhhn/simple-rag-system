"""
Startup utilities for prewarming and initializing embedding models.

This module provides utilities to optimize application startup by
preloading commonly used models into memory.
"""
from typing import List, Optional
from pathlib import Path

from src.core.logging import get_logger
from src.core.config import get_config
from src.embedding.model_manager import ModelManager


logger = get_logger(__name__)


def prewarm_embedding_models(model_names: Optional[List[str]] = None) -> None:
    """
    Prewarm BGE-M3 embedding model by loading it into memory.
    
    This function should be called during application startup to
    load BGE-M3 model into memory, reducing first-request latency.
    
    Args:
        model_names: List of model names to preload (will only use BGE-M3). 
                    If None, loads default model.
        
    Example:
        >>> # Load default BGE-M3 model
        >>> prewarm_embedding_models()
    """
    config = get_config()
    
    if model_names is None:
        model_names = [config.embedding.model_name]
    else:
        # Only keep BGE-M3 models
        model_names = ["BAAI/bge-m3"]
    
    logger.info("Prewarming BGE-M3 embedding model", models=model_names)
    
    try:
        manager = ModelManager.get_instance()
        manager.preload_models(model_names)
        
        stats = manager.get_stats()
        logger.info(
            "BGE-M3 prewarming completed",
            loaded=stats["total_models"],
            references=stats["total_references"]
        )
        
    except Exception as e:
        logger.error("Failed to prewarm BGE-M3 model", error=str(e), models=model_names)
        # Don't raise exception - allow application to start without prewarming


def get_model_startup_time(model_name: str = "BAAI/bge-m3") -> float:
    """
    Measure the time it takes to load BGE-M3 model.
    
    Useful for benchmarking and optimizing startup performance.
    
    Args:
        model_name: Name of the model to benchmark (default: BAAI/bge-m3)
        
    Returns:
        Loading time in seconds
        
    Example:
        >>> time = get_model_startup_time()
        >>> print(f"Loading time: {time:.2f}s")
    """
    import time
    from src.embedding import ModelLoader
    
    # Always use BGE-M3
    model_name = "BAAI/bge-m3"
    
    # First unload if already loaded
    manager = ModelManager.get_instance()
    manager.unload_model(model_name)
    
    # Measure loading time
    start_time = time.time()
    
    try:
        config = get_config()
        loader = ModelLoader(cache_dir=config.storage.model_cache_path)
        loader.load_model(model_name)
        
        elapsed_time = time.time() - start_time
        logger.info("BGE-M3 startup time measured", model=model_name, time=f"{elapsed_time:.2f}s")
        
        return elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            "Failed to measure BGE-M3 startup time",
            model=model_name,
            error=str(e),
            elapsed_time=f"{elapsed_time:.2f}s"
        )
        raise


def cache_model_info(model_names: Optional[List[str]] = None, output_file: Optional[Path] = None) -> None:
    """
    Cache information about BGE-M3 model for quick reference.
    
    This function loads BGE-M3 model and caches its metadata (dimensions,
    sizes, etc.) to a file for quick reference without loading the model.
    
    Args:
        model_names: List of model names to analyze (will only use BGE-M3)
        output_file: Path to save cache file (default: data/cache/model_info.json)
        
    Example:
        >>> cache_model_info()
    """
    import json
    from src.embedding import ModelLoader
    
    if output_file is None:
        config = get_config()
        output_file = config.storage.cache_path / "model_info.json"
    
    # Always use BGE-M3
    model_names = ["BAAI/bge-m3"]
    
    logger.info("Caching BGE-M3 model information", models=model_names)
    
    model_info = {}
    
    try:
        manager = ModelManager.get_instance()
        
        for model_name in model_names:
            logger.info("Analyzing BGE-M3 model", model=model_name)
            
            try:
                model = manager.get_model(model_name)
                dimension = model.get_sentence_embedding_dimension()
                
                model_info[model_name] = {
                    "dimension": dimension,
                    "max_seq_length": model.max_seq_length if hasattr(model, 'max_seq_length') else 8192,
                }
                
                logger.info("BGE-M3 model analyzed", model=model_name, dimension=dimension)
                
            except Exception as e:
                logger.error("Failed to analyze BGE-M3 model", model=model_name, error=str(e))
                model_info[model_name] = {"error": str(e)}
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("BGE-M3 model information cached", file=str(output_file))
        
    except Exception as e:
        logger.error("Failed to cache BGE-M3 model information", error=str(e))
        raise


def get_cached_model_info(output_file: Optional[Path] = None) -> dict:
    """
    Load cached model information from file.
    
    Args:
        output_file: Path to cache file (default: data/cache/model_info.json)
        
    Returns:
        Dictionary with model information
        
    Example:
        >>> info = get_cached_model_info()
        >>> print(info["sentence-transformers/all-MiniLM-L6-v2"]["dimension"])
    """
    import json
    
    if output_file is None:
        config = get_config()
        output_file = config.storage.cache_path / "model_info.json"
    
    if not output_file.exists():
        logger.warning("Model info cache not found", file=str(output_file))
        return {}
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        logger.info("Model information loaded from cache", file=str(output_file))
        return model_info
        
    except Exception as e:
        logger.error("Failed to load model info cache", error=str(e))
        return {}
