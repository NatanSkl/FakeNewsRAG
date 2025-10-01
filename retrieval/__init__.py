"""
Retrieval module for the FakeNewsRAG system.

This module contains core retrieval functions and testing utilities.
"""

from .retrieval import (
    load_store,
    encode,
    hybrid_once,
    claimify,
    retrieve_evidence,
    RetrievalConfig,
    Store
)

__all__ = [
    'load_store',
    'encode', 
    'hybrid_once',
    'claimify',
    'retrieve_evidence',
    'RetrievalConfig',
    'Store'
]
