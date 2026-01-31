#!/usr/bin/env python3
"""
Quick test script for BGE-M3 embedding model.

This script tests that the BGE-M3 model is working correctly.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging import get_logger
from src.embedding import BGEM3Model


logger = get_logger(__name__)


def test_basic_encoding():
    """Test basic encoding functionality."""
    print("\n" + "="*70)
    print("Test 1: Basic Encoding")
    print("="*70 + "\n")

    model = BGEM3Model()

    # Test Vietnamese
    text_vi = "Xin chào, đây là văn bản tiếng Việt"
    embedding_vi = model.encode_single(text_vi)
    print(f"✓ Vietnamese text: {text_vi}")
    print(f"  Dimension: {len(embedding_vi)}")
    print(f"  Sample values: {embedding_vi[:5]}")

    # Test English
    text_en = "Hello world, this is English text"
    embedding_en = model.encode_single(text_en)
    print(f"\n✓ English text: {text_en}")
    print(f"  Dimension: {len(embedding_en)}")
    print(f"  Sample values: {embedding_en[:5]}")

    return True


def test_batch_encoding():
    """Test batch encoding functionality."""
    print("\n" + "="*70)
    print("Test 2: Batch Encoding")
    print("="*70 + "\n")

    model = BGEM3Model()

    texts = [
        "Text 1 in Vietnamese",
        "Text 2 in English",
        "Text 3 in mixed language"
    ]

    embeddings = model.encode(texts, batch_size=3)
    print(f"✓ Encoded {len(texts)} texts")
    for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
        print(f"  Text {i}: {text}")
        print(f"    Dimension: {len(emb)}")

    return True


def test_similarity():
    """Test similarity computation."""
    print("\n" + "="*70)
    print("Test 3: Similarity Computation")
    print("="*70 + "\n")

    model = BGEM3Model()

    # Similar texts (Vietnamese and English)
    text1 = "Học máy"
    text2 = "Machine learning"

    emb1 = model.encode_single(text1)
    emb2 = model.encode_single(text2)

    similarity = model.compute_similarity(emb1, emb2)
    print(f"✓ Text 1: {text1}")
    print(f"  Text 2: {text2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  → High similarity indicates good multilingual understanding")

    return True


def test_long_context():
    """Test long context handling."""
    print("\n" + "="*70)
    print("Test 4: Long Context Handling")
    print("="*70 + "\n")

    model = BGEM3Model()

    # Create a long text
    long_text = " ".join(["Xin chào"] * 1000)
    print(f"✓ Text length: {len(long_text)} characters")

    embedding = model.encode_single(long_text)
    print(f"  Dimension: {len(embedding)}")
    print(f"  → Long context handled successfully")

    return True


def test_model_info():
    """Test model information."""
    print("\n" + "="*70)
    print("Test 5: Model Information")
    print("="*70 + "\n")

    model = BGEM3Model()
    print(f"✓ Model name: {model.model_name}")
    print(f"  Dimension: {model.dimension}")
    print(f"  Max length: {model.max_length}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BGE-M3 Model Test Suite")
    print("="*70)

    tests = [
        ("Basic Encoding", test_basic_encoding),
        ("Batch Encoding", test_batch_encoding),
        ("Similarity Computation", test_similarity),
        ("Long Context Handling", test_long_context),
        ("Model Information", test_model_info),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name}: FAILED")
            print(f"  Error: {e}")
            logger.error(f"Test failed: {test_name}", error=str(e), exc_info=True)

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ All tests passed! BGE-M3 model is working correctly.\n")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please check logs.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
