"""
Embedding model implementations.

This module provides the BGE-M3 embedding model - the primary and only
embedding model used in the system. BGE-M3 offers superior performance,
long context support (8192 tokens), and multilingual capabilities.
"""

from .bgem3 import BGEM3Model

__all__ = [
    "BGEM3Model",
]
