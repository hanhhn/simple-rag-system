"""
Embedding cache for improving performance.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from src.core.logging import get_logger
from src.core.exceptions import EmbeddingCacheError


logger = get_logger(__name__)


class EmbeddingCache:
    """
    Cache for storing embeddings to avoid recomputation.
    
    This cache stores text-to-embeddings mappings, allowing the system
    to reuse previously computed embeddings for identical text inputs.
    
    Attributes:
        cache_dir: Directory to store cache files
        cache_file: Path to the cache file
        cache: In-memory cache dictionary
        enabled: Whether caching is enabled
        
    Example:
        >>> cache = EmbeddingCache(cache_dir="./data/cache")
        >>> embedding = cache.get("Hello world")
        >>> if embedding is None:
        ...     # Compute embedding
        ...     cache.set("Hello world", [0.1, 0.2, 0.3])
    """
    
    def __init__(self, cache_dir: Path | str, enabled: bool = True) -> None:
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether to enable caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "embeddings_cache.json"
        self.enabled = enabled
        self.cache: dict = {}
        
        # Load existing cache if enabled
        if self.enabled and self.cache_file.exists():
            self._load()
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            MD5 hash of the text as cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache if it exists.
        
        Args:
            text: Text to look up in cache
            
        Returns:
            Cached embedding vector or None if not found
        """
        if not self.enabled:
            return None
        
        key = self._get_cache_key(text)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store an embedding in the cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector to cache
            
        Example:
            >>> cache.set("Hello world", [0.1, 0.2, 0.3])
        """
        if not self.enabled:
            return
        
        key = self._get_cache_key(text)
        self.cache[key] = embedding
    
    def get_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            texts: List of texts to look up
            
        Returns:
            List of cached embeddings (None for cache misses)
        """
        return [self.get(text) for text in texts]
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of texts that were embedded
            embeddings: List of embedding vectors to cache
            
        Example:
            >>> cache.set_batch(["text1", "text2"], [[0.1], [0.2]])
        """
        if not self.enabled:
            return
        
        if len(texts) != len(embeddings):
            logger.warning(
                "Batch size mismatch in cache set",
                texts_count=len(texts),
                embeddings_count=len(embeddings)
            )
            return
        
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self) -> None:
        """
        Clear all cached embeddings.
        
        Example:
            >>> cache.clear()
        """
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def save(self) -> None:
        """
        Save cache to disk.
        
        Example:
            >>> cache.save()
        """
        if not self.enabled:
            return
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
            logger.info("Embedding cache saved", cache_file=str(self.cache_file))
        except Exception as e:
            raise EmbeddingCacheError(
                f"Failed to save cache: {str(e)}",
                details={"cache_file": str(self.cache_file), "error": str(e)}
            )
    
    def _load(self) -> None:
        """
        Load cache from disk.
        
        Raises:
            EmbeddingCacheError: If cache file cannot be loaded
        """
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(
                "Embedding cache loaded",
                cache_file=str(self.cache_file),
                entries=len(self.cache)
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "Cache file is corrupted, starting with empty cache",
                cache_file=str(self.cache_file),
                error=str(e)
            )
            self.cache = {}
        except Exception as e:
            raise EmbeddingCacheError(
                f"Failed to load cache: {str(e)}",
                details={"cache_file": str(self.cache_file), "error": str(e)}
            )
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.enabled,
            "entries": len(self.cache),
            "cache_file": str(self.cache_file),
            "cache_size_bytes": self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
