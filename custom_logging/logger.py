"""
Core logging functionality for the FakeNewsRAG system.

This module provides logging setup, configuration, and utility functions
that are shared across different components.
"""

import logging
import torch
from typing import Optional


def setup_logging(
    log_file: str = "output.log",
    log_level: int = logging.DEBUG,
    include_console: bool = True
) -> None:
    """
    Configure logging to output to a log file with timestamps and appropriate levels.
    
    Args:
        log_file: Path to the log file (default: "output.log")
        log_level: Logging level (default: logging.DEBUG)
        include_console: Whether to also output to console (default: True)
    """
    handlers = [logging.FileHandler(log_file)]
    
    if include_console:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - output will be written to {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system information for debugging purposes.
    
    Args:
        logger: Optional logger instance. If None, uses the root logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # GPU information
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.debug(f"GPU memory: {gpu_memory:.1f} GB")
            logger.debug(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            logger.debug("CUDA not available")
        
    except Exception as e:
        logger.warning(f"Could not retrieve system information: {e}")


def log_gpu_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log GPU-specific information.
    
    Args:
        logger: Optional logger instance. If None, uses the root logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        import faiss
        ngpu = faiss.get_num_gpus()
        if ngpu > 0:
            logger.info(f"FAISS GPU support: {ngpu} GPU(s) available")
            for i in range(ngpu):
                props = faiss.get_device_properties(i)
                logger.debug(f"GPU {i}: {props.name}, {props.major}.{props.minor}")
        else:
            logger.info("FAISS GPU support not available")
    except Exception as e:
        logger.warning(f"Could not retrieve GPU information: {e}")
