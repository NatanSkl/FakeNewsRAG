#!/usr/bin/env python3
"""
Example usage of the RAG Executor

This script demonstrates how to use the RAG executor to run the RAG pipeline
on CSV files with various configurations.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.rag_executor import RAGExecutor
from retrieval.retrieval_v3 import RetrievalConfig


def main():
    """Example usage of the RAG executor."""
    
    # Initialize the executor
    executor = RAGExecutor(
        llm_url="http://127.0.0.1:8010",
        store_path="/StudentData/index"
    )
    
    # Initialize connections
    executor.initialize()
    
    # Create retrieval configuration
    retrieval_config = RetrievalConfig(
        k=8,  # Number of evidence chunks to retrieve
        ce_model_name=None,  # Cross-encoder model name (optional)
        diversity_type=None,  # Diversity method (optional)
        verbose=False
    )
    
    # Run RAG pipeline on a CSV file
    output_path = executor.run_rag_pipeline(
        csv_path="/StudentData/preprocessed/val_sampled.csv",
        output_dir="experiments",
        retrieval_config=retrieval_config,
        prompt_type=0,  # Prompt type for summarization and classification
        naming_convention="fake_reliable",  # Naming convention for labels
        limit=15  # Limit to 5 articles for testing
    )
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
