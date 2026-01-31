#!/usr/bin/env python
"""
Test script to verify Granite embedding model implementation.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer

from src.core.logging import get_logger
from src.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


def test_direct_sentence_transformers():
    """Test direct embedding generation using sentence_transformers library."""
    logger.info("=" * 60)
    logger.info("Testing Direct Sentence-Transformers Granite Model")
    logger.info("=" * 60)

    try:
        # Load model directly
        logger.info("Loading Granite model with sentence-transformers...")
        model = SentenceTransformer("ibm-granite/granite-embedding-small-english-r2")
        logger.info(f"Model loaded successfully")
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")

        # Test single embedding
        test_text = "Hello world"
        logger.info(f"Testing single embedding for: '{test_text}'")
        embedding = model.encode(test_text)
        logger.info(f"✓ Single embedding shape: {embedding.shape}")

        # Test batch embedding
        texts = ["Hello world", "Natural language processing"]
        logger.info(f"Testing batch embedding for {len(texts)} texts...")
        embeddings = model.encode(texts)
        logger.info(f"✓ Batch embeddings shape: {embeddings.shape}")
        assert embeddings.shape == (2, 384), f"Expected shape (2, 384), got {embeddings.shape}"

        # Verify embedding values are valid
        assert embeddings.shape[1] == 384, f"Expected 384 dimensions, got {embeddings.shape[1]}"
        assert not np.isnan(embeddings).any(), "Found NaN values in embeddings"
        logger.info("✓ Embeddings are valid (no NaN values)")

        logger.info("=" * 60)
        logger.info("Direct sentence-transformers test passed! ✓")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Direct test failed: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        logger.error("Direct test failed! ✗")
        logger.error("=" * 60)
        return False


def test_granite_model():
    """Test Granite embedding model."""
    logger.info("=" * 60)
    logger.info("Testing Granite Embedding Model")
    logger.info("=" * 60)

    try:
        # Initialize service
        logger.info("Initializing EmbeddingService...")
        service = EmbeddingService()
        logger.info(f"Model name: {service.get_model_name()}")

        # Test dimension
        logger.info(f"Embedding dimension: {service.get_dimension()}")
        assert service.get_dimension() == 384, f"Expected dimension 384, got {service.get_dimension()}"
        logger.info("✓ Dimension test passed")

        # Test single embedding
        test_text = "This is a test sentence for Granite embedding model."
        logger.info(f"Testing single embedding for: '{test_text[:50]}...'")
        embedding = service.generate_embedding(test_text)
        assert len(embedding) == 384, f"Expected embedding length 384, got {len(embedding)}"
        logger.info(f"✓ Single embedding test passed (length: {len(embedding)})")

        # Test batch embedding
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence for batch processing."
        ]
        logger.info(f"Testing batch embedding for {len(test_texts)} texts...")
        embeddings = service.generate_embeddings(test_texts, batch_size=2)
        assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
        for i, emb in enumerate(embeddings):
            assert len(emb) == 384, f"Embedding {i} has wrong length: {len(emb)}"
        logger.info("✓ Batch embedding test passed")

        # Test similarity computation
        logger.info("Testing similarity computation...")
        similarity = service.compute_similarity(embeddings[0], embeddings[1])
        assert 0.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"
        logger.info(f"✓ Similarity test passed (score: {similarity:.4f})")

        # Test cache stats
        stats = service.get_cache_stats()
        logger.info(f"Cache stats: {stats}")

        logger.info("=" * 60)
        logger.info("All tests passed! ✓")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        logger.error("Test failed! ✗")
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    # Run direct sentence-transformers test first
    direct_success = test_direct_sentence_transformers()
    print()

    # Run EmbeddingService test
    service_success = test_granite_model()
    print()

    # Exit with failure if either test failed
    success = direct_success and service_success
    if success:
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! ✓✓")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("SOME TESTS FAILED! ✗")
        logger.error("=" * 60)

    sys.exit(0 if success else 1)
