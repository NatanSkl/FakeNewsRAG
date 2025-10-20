"""
Generic Evaluator

Evaluates classification results from CSV files and calculates metrics.
"""

import time
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class EvaluationMetrics:
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
    article_id: str
    true_label: str
    predicted_label: str
    confidence: float
    processing_time: float
    correct: bool
    method: str  # "llama" or "rag"


class Evaluator:
    """Generic evaluator for classification results from CSV files."""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("CLASSIFICATION EVALUATION SYSTEM")
        print("="*80)
        print("Analyzes classification results and calculates metrics")
        print("All output in English")
        print()
    
    def evaluate_from_csv(self, csv_path: str) -> Tuple[List[ArticleResult], EvaluationMetrics]:
        """Load results from CSV file and evaluate them."""
        print(f"\nEVALUATING RESULTS FROM: {csv_path}")
        print("-"*80)
        
        # Load results from CSV
        results = self._load_results_from_csv(csv_path)
        
        # Evaluate
        metrics = self.evaluate_results(results)
        
        # Save metrics
        self._save_metrics(metrics, csv_path)
        
        return results, metrics
    
    def evaluate_results(self, results: List[ArticleResult]) -> EvaluationMetrics:
        """Evaluate classification results and calculate metrics."""
        print(f"Evaluating {len(results)} classification results...")
        
        # Calculate total time
        total_time = sum(r.processing_time for r in results)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, total_time)
        
        print(f"\nClassification Evaluation Results:")
        print(f"  Accuracy: {metrics.accuracy:.1%}")
        print(f"  Precision: {metrics.precision:.1%}")
        print(f"  Recall: {metrics.recall:.1%}")
        print(f"  F1 Score: {metrics.f1_score:.1%}")
        print(f"  Avg time: {metrics.avg_time_per_article:.2f}s per article")
        
        return metrics
    
    def _save_metrics(self, metrics: EvaluationMetrics, csv_path: str):
        """Save metrics to JSON file in metrics directory."""
        # Generate metrics filename based on input CSV
        input_filename = Path(csv_path).stem  # e.g., "val_sampled_llm_baseline"
        metrics_filename = f"{input_filename}_metrics.json"
        metrics_path = self.metrics_dir / metrics_filename
        
        # Create metrics data
        metrics_data = {
            "metrics": metrics.to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_csv": str(csv_path)
        }
        
        # Save to JSON
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_path}")
    
    def _load_results_from_csv(self, csv_path: str) -> List[ArticleResult]:
        """Load classification results from CSV file."""
        df = pd.read_csv(csv_path)
        
        results = []
        for _, row in df.iterrows():
            result = ArticleResult(
                article_id=str(row.get('article_id', '')),
                true_label=str(row.get('true_label', '')),
                predicted_label=str(row.get('predicted_label', '')),
                confidence=float(row.get('confidence', 0.0)),
                processing_time=float(row.get('processing_time', 0.0)),
                correct=bool(row.get('correct', False)),
                method="llama"
            )
            results.append(result)
        
        print(f"Loaded {len(results)} results from CSV")
        return results
    
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
    
    def save_results(self, results: List[ArticleResult], metrics: EvaluationMetrics, filename: str = "llm_results.json"):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        
        data = {
            "metrics": metrics.to_dict(),
            "results": [asdict(result) for result in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate classification results from CSV")
    parser.add_argument("csv_path", help="Path to CSV file with classification results")
    parser.add_argument("--metrics-dir", default="metrics", help="Directory to save metrics")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.metrics_dir)
    
    # Evaluate results
    results, metrics = evaluator.evaluate_from_csv(args.csv_path)
    
    print(f"\nEvaluation complete! Metrics saved to: {args.metrics_dir}")


if __name__ == "__main__":
    main()
