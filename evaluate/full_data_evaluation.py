"""
Full Data Evaluation Script

Main evaluation script that uses the separated evaluator classes.
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the separated evaluator classes
from evaluate.only_llm_evaluator import OnlyLLMEvaluator
from evaluate.rag_evaluator import RAGEvaluator
from evaluate.comparison import ResultsVisualizer, ComparisonReporter


def load_test_data(data_path: str, max_articles: int = None) -> List[Dict]:
    """Load test data from CSV file."""
    print(f"Loading test data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if max_articles:
        df = df.head(max_articles)
        print(f"Limited to {max_articles} articles")
    
    # Convert to list of dictionaries
    articles = []
    for _, row in df.iterrows():
        article = {
            'id': str(row.get('id', '')),
            'title': str(row.get('title', '')),
            'content': str(row.get('content', '')),
            'label': str(row.get('label', '')).lower()
        }
        articles.append(article)
    
    print(f"Loaded {len(articles)} articles")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return articles


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="RAG vs Llama Evaluation")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--store", required=True, help="Path to vector store")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to test")
    parser.add_argument("--output", default="evaluate/results", help="Output directory")
    parser.add_argument("--llama-only", action="store_true", help="Run only Llama evaluation")
    parser.add_argument("--rag-only", action="store_true", help="Run only RAG evaluation")
    
    args = parser.parse_args()
    
    # Load test data
    articles = load_test_data(args.data, args.max_articles)
    
    # Initialize output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llama_results = None
    llama_metrics = None
    rag_results = None
    rag_metrics = None
    
    # Run Llama evaluation
    if not args.rag_only:
        print("\n" + "="*80)
        print("RUNNING LLAMA EVALUATION")
        print("="*80)
        
        llama_evaluator = OnlyLLMEvaluator(str(output_dir))
        llama_evaluator.initialize()
        llama_results, llama_metrics = llama_evaluator.run_llm_baseline(articles)
        llama_evaluator.save_results(llama_results, llama_metrics)
    
    # Run RAG evaluation
    if not args.llama_only:
        print("\n" + "="*80)
        print("RUNNING RAG EVALUATION")
        print("="*80)
        
        rag_evaluator = RAGEvaluator(args.store, str(output_dir))
        rag_evaluator.initialize()
        rag_results, rag_metrics = rag_evaluator.run_rag_pipeline(articles)
        rag_evaluator.save_results(rag_results, rag_metrics)
    
    # Generate comparison if both evaluations were run
    if llama_results is not None and rag_results is not None:
        print("\n" + "="*80)
        print("GENERATING COMPARISON")
        print("="*80)
        
        # Create visualizations
        visualizer = ResultsVisualizer(output_dir)
        visualizer.plot_comparison(llama_metrics, rag_metrics)
        
        # Generate comparison report
        reporter = ComparisonReporter(output_dir)
        reporter.compare_and_report(llama_results, llama_metrics, rag_results, rag_metrics)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()