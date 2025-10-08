"""
Complete RAG vs Llama Evaluation System

This script provides a comprehensive comparison between:
1. RAG (Retrieval-Augmented Generation) pipeline
2. Llama 3.2 3B baseline LLM

Features:
- Batch processing for large datasets
- Real performance metrics (Accuracy, Precision, Recall, F1)
- Visualization with graphs
- Detailed comparison reports
- Professional output in English only

No emojis, production-ready code.
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG components
from pipeline.rag_pipeline import classify_article_rag, RAGOutput
from retrieval import load_store, RetrievalConfig
from common.llm_client import Llama


@dataclass
class EvaluationMetrics:
    """Metrics for evaluation results."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    avg_time_per_article: float
    total_time: float
    total_articles: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArticleResult:
    """Result for a single article classification."""
    article_id: str
    true_label: str
    predicted_label: str
    confidence: float
    processing_time: float
    correct: bool
    method: str  # "llama" or "rag"


class ResultsVisualizer:
    """Generate visualization graphs for evaluation results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_comparison(self, llama_metrics: EvaluationMetrics, rag_metrics: EvaluationMetrics):
        """Create comparison plots between Llama and RAG."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            print("WARNING: matplotlib not installed, skipping graphs")
            return
        
        # Accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RAG vs Llama Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        llama_values = [llama_metrics.accuracy, llama_metrics.precision, 
                       llama_metrics.recall, llama_metrics.f1_score]
        rag_values = [rag_metrics.accuracy, rag_metrics.precision, 
                     rag_metrics.recall, rag_metrics.f1_score]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, llama_values, width, label='Llama', color='#3498db')
        axes[0, 0].bar(x + width/2, rag_values, width, label='RAG', color='#2ecc71')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Processing time comparison
        time_data = ['Llama', 'RAG']
        time_values = [llama_metrics.avg_time_per_article, rag_metrics.avg_time_per_article]
        colors = ['#3498db', '#2ecc71']
        
        axes[0, 1].bar(time_data, time_values, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Average Processing Time per Article')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, value) in enumerate(zip(time_data, time_values)):
            axes[0, 1].text(i, value, f'{value:.2f}s', ha='center', va='bottom')
        
        # 3. Confusion matrices
        # Llama confusion matrix
        llama_cm = np.array([
            [llama_metrics.true_negatives, llama_metrics.false_positives],
            [llama_metrics.false_negatives, llama_metrics.true_positives]
        ])
        
        im1 = axes[1, 0].imshow(llama_cm, cmap='Blues', alpha=0.7)
        axes[1, 0].set_title('Llama Confusion Matrix')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Reliable', 'Fake'])
        axes[1, 0].set_yticklabels(['Reliable', 'Fake'])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 0].text(j, i, str(llama_cm[i, j]), 
                              ha='center', va='center', color='black', fontsize=14)
        
        # RAG confusion matrix
        rag_cm = np.array([
            [rag_metrics.true_negatives, rag_metrics.false_positives],
            [rag_metrics.false_negatives, rag_metrics.true_positives]
        ])
        
        im2 = axes[1, 1].imshow(rag_cm, cmap='Greens', alpha=0.7)
        axes[1, 1].set_title('RAG Confusion Matrix')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Reliable', 'Fake'])
        axes[1, 1].set_yticklabels(['Reliable', 'Fake'])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, str(rag_cm[i, j]), 
                              ha='center', va='center', color='black', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'comparison_graphs.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nGraphs saved to: {output_path}")
        
        return str(output_path)


class RAGvsLlamaEvaluator:
    """Main evaluator for comparing RAG and Llama."""
    
    def __init__(self, store_path: str, output_dir: str = "evaluate/results"):
        self.store_path = store_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = None
        self.store = None
        
        print("="*80)
        print("RAG vs LLAMA EVALUATION SYSTEM")
        print("="*80)
        print("Professional evaluation with real metrics and graphs")
        print("All output in English")
        print()
    
    def initialize(self):
        """Initialize LLM and store."""
        print("INITIALIZATION")
        print("-"*80)
        
        # Load Llama model
        print("Step 1/2: Loading Llama 3.2 3B model...")
        try:
            from llama_cpp import Llama as LlamaCpp
            self.llm = LlamaCpp(
                model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                n_ctx=1024,
                n_gpu_layers=0,
                verbose=False
            )
            print("Llama model loaded successfully")
        except Exception as e:
            print(f"ERROR loading Llama: {e}")
            raise
        
        # Load store
        print("Step 2/2: Loading vector store...")
        try:
            self.store = load_store(self.store_path, verbose=False)
            print(f"Store loaded successfully: {self.store.index.ntotal} vectors")
        except Exception as e:
            print(f"ERROR loading store: {e}")
            raise
        
        print("\nInitialization complete!")
        print("="*80)
    
    def test_llama_baseline(self, articles: List[Dict]) -> Tuple[List[ArticleResult], EvaluationMetrics]:
        """Test Llama baseline on articles."""
        print("\nTEST 1: LLAMA BASELINE")
        print("-"*80)
        print(f"Testing on {len(articles)} articles...")
        
        results = []
        start_total = time.time()
        
        for i, article in enumerate(articles, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(articles)} articles processed...")
            
            start_time = time.time()
            
            # Create prompt
            prompt = f"""Classify this news article as FAKE or RELIABLE.

Title: {article['title']}
Content: {article['content'][:500]}

Classification (respond with only FAKE or RELIABLE):"""
            
            try:
                response = self.llm(
                    prompt,
                    max_tokens=10,
                    temperature=0.1,
                    stop=["\n"]
                )
                
                response_text = response['choices'][0]['text'].strip().upper()
                
                # Parse prediction
                if "FAKE" in response_text:
                    prediction = "fake"
                elif "RELIABLE" in response_text or "CREDIBLE" in response_text:
                    prediction = "reliable"
                else:
                    prediction = "reliable"  # default
                
                confidence = 0.8  # Llama doesn't provide confidence
                
            except Exception as e:
                print(f"  Error processing article {i}: {e}")
                prediction = "reliable"
                confidence = 0.5
            
            elapsed = time.time() - start_time
            
            result = ArticleResult(
                article_id=article.get('id', f'article_{i}'),
                true_label=article['label'],
                predicted_label=prediction,
                confidence=confidence,
                processing_time=elapsed,
                correct=(prediction == article['label']),
                method="llama"
            )
            results.append(result)
        
        total_time = time.time() - start_total
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, total_time)
        
        print(f"\nLlama Baseline Results:")
        print(f"  Accuracy: {metrics.accuracy:.1%}")
        print(f"  Precision: {metrics.precision:.1%}")
        print(f"  Recall: {metrics.recall:.1%}")
        print(f"  F1 Score: {metrics.f1_score:.1%}")
        print(f"  Avg time: {metrics.avg_time_per_article:.2f}s per article")
        
        return results, metrics
    
    def test_rag_pipeline(self, articles: List[Dict]) -> Tuple[List[ArticleResult], EvaluationMetrics]:
        """Test RAG pipeline on articles."""
        print("\nTEST 2: RAG PIPELINE")
        print("-"*80)
        print(f"Testing on {len(articles)} articles...")
        
        results = []
        start_total = time.time()
        
        config = RetrievalConfig(k=10, ce_model=None, diversity_type=None, verbose=False)
        
        for i, article in enumerate(articles, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(articles)} articles processed...")
            
            start_time = time.time()
            
            try:
                # Use RAG pipeline
                rag_result = classify_article_rag(
                    article_title=article['title'],
                    article_content=article['content'],
                    store=self.store,
                    llm=self.llm,
                    retrieval_config=config,
                    verbose=False
                )
                
                prediction = rag_result.classification.prediction
                confidence = rag_result.classification.confidence
                
            except Exception as e:
                print(f"  Error processing article {i}: {e}")
                prediction = "reliable"
                confidence = 0.5
            
            elapsed = time.time() - start_time
            
            result = ArticleResult(
                article_id=article.get('id', f'article_{i}'),
                true_label=article['label'],
                predicted_label=prediction,
                confidence=confidence,
                processing_time=elapsed,
                correct=(prediction == article['label']),
                method="rag"
            )
            results.append(result)
        
        total_time = time.time() - start_total
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, total_time)
        
        print(f"\nRAG Pipeline Results:")
        print(f"  Accuracy: {metrics.accuracy:.1%}")
        print(f"  Precision: {metrics.precision:.1%}")
        print(f"  Recall: {metrics.recall:.1%}")
        print(f"  F1 Score: {metrics.f1_score:.1%}")
        print(f"  Avg time: {metrics.avg_time_per_article:.2f}s per article")
        
        return results, metrics
    
    def _calculate_metrics(self, results: List[ArticleResult], total_time: float) -> EvaluationMetrics:
        """Calculate evaluation metrics from results."""
        # Count TP, FP, TN, FN
        tp = sum(1 for r in results if r.true_label == "fake" and r.predicted_label == "fake")
        fp = sum(1 for r in results if r.true_label == "reliable" and r.predicted_label == "fake")
        tn = sum(1 for r in results if r.true_label == "reliable" and r.predicted_label == "reliable")
        fn = sum(1 for r in results if r.true_label == "fake" and r.predicted_label == "reliable")
        
        # Calculate metrics
        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            avg_time_per_article=avg_time,
            total_time=total_time,
            total_articles=len(results)
        )
    
    def compare_and_report(
        self, 
        llama_results: List[ArticleResult],
        llama_metrics: EvaluationMetrics,
        rag_results: List[ArticleResult],
        rag_metrics: EvaluationMetrics
    ):
        """Generate comparison report and visualizations."""
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80)
        
        # Console output
        print("\nPerformance Comparison:")
        print("-"*80)
        print(f"{'Metric':<20} {'Llama':<15} {'RAG':<15} {'Difference':<15}")
        print("-"*80)
        print(f"{'Accuracy':<20} {llama_metrics.accuracy:<15.1%} {rag_metrics.accuracy:<15.1%} {(rag_metrics.accuracy - llama_metrics.accuracy):+.1%}")
        print(f"{'Precision':<20} {llama_metrics.precision:<15.1%} {rag_metrics.precision:<15.1%} {(rag_metrics.precision - llama_metrics.precision):+.1%}")
        print(f"{'Recall':<20} {llama_metrics.recall:<15.1%} {rag_metrics.recall:<15.1%} {(rag_metrics.recall - llama_metrics.recall):+.1%}")
        print(f"{'F1 Score':<20} {llama_metrics.f1_score:<15.1%} {rag_metrics.f1_score:<15.1%} {(rag_metrics.f1_score - llama_metrics.f1_score):+.1%}")
        print(f"{'Avg Time (s)':<20} {llama_metrics.avg_time_per_article:<15.2f} {rag_metrics.avg_time_per_article:<15.2f} {(rag_metrics.avg_time_per_article - llama_metrics.avg_time_per_article):+.2f}")
        
        # Analysis
        print("\nAnalysis:")
        print("-"*80)
        
        if rag_metrics.accuracy > llama_metrics.accuracy:
            improvement = (rag_metrics.accuracy - llama_metrics.accuracy) / llama_metrics.accuracy * 100
            print(f"RAG improves accuracy by {improvement:.1f}% over Llama baseline")
        else:
            print("Llama baseline performs better on this dataset")
        
        speed_ratio = rag_metrics.avg_time_per_article / llama_metrics.avg_time_per_article
        print(f"RAG is {speed_ratio:.1f}x slower than Llama (retrieval overhead)")
        
        # Save JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'llama': {
                'metrics': llama_metrics.to_dict(),
                'results': [asdict(r) for r in llama_results]
            },
            'rag': {
                'metrics': rag_metrics.to_dict(),
                'results': [asdict(r) for r in rag_results]
            },
            'comparison': {
                'accuracy_improvement': rag_metrics.accuracy - llama_metrics.accuracy,
                'speed_ratio': speed_ratio,
                'better_method': 'RAG' if rag_metrics.accuracy > llama_metrics.accuracy else 'Llama'
            }
        }
        
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Generate graphs
        print("\nGenerating visualization graphs...")
        visualizer = ResultsVisualizer(self.output_dir)
        graph_path = visualizer.plot_comparison(llama_metrics, rag_metrics)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved in: {self.output_dir}")
        print(f"  - evaluation_report.json (detailed metrics)")
        print(f"  - comparison_graphs.png (visual comparison)")


def load_test_data(csv_path: str, max_articles: int = None) -> List[Dict]:
    """Load test data from CSV file."""
    print(f"\nLoading test data from: {csv_path}")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['label', 'title', 'content']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Filter and convert
    df = df[df['label'].isin(['fake', 'reliable', 'credible'])]
    df['label'] = df['label'].replace('credible', 'reliable')
    
    if max_articles:
        df = df.head(max_articles)
    
    articles = df.to_dict('records')
    
    print(f"Loaded {len(articles)} articles:")
    print(f"  Fake: {sum(1 for a in articles if a['label'] == 'fake')}")
    print(f"  Reliable: {sum(1 for a in articles if a['label'] == 'reliable')}")
    
    return articles


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG vs Llama Evaluation")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--store", required=True, help="Path to vector store")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to test")
    parser.add_argument("--output", default="evaluate/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load test data
    articles = load_test_data(args.data, args.max_articles)
    
    # Initialize evaluator
    evaluator = RAGvsLlamaEvaluator(args.store, args.output)
    evaluator.initialize()
    
    # Run tests
    llama_results, llama_metrics = evaluator.test_llama_baseline(articles)
    rag_results, rag_metrics = evaluator.test_rag_pipeline(articles)
    
    # Generate comparison
    evaluator.compare_and_report(llama_results, llama_metrics, rag_results, rag_metrics)


if __name__ == "__main__":
    main()
