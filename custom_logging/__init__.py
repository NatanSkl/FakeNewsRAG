"""
Shared logging utilities for the FakeNewsRAG system.

This module provides centralized logging configuration and utilities
that can be used across different components of the system.
"""

from .logger import setup_logging, get_logger, log_system_info

__all__ = [
    'setup_logging',
    'get_logger', 
    'log_system_info'
]
