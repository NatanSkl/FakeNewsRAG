"""
Retrieval module for the FakeNewsRAG system.

This module contains core retrieval functions and testing utilities.
"""

from .retrieval_v3 import (
    load_store,
    retrieve_evidence,
    Store,
    RetrievalConfig
)

__all__ = [
    'load_store',
    'retrieve_evidence',
    'Store',
    'RetrievalConfig'
]
