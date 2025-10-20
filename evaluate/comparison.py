"""
Comparison and Visualization Module

Handles comparison between different evaluation results and creates visualizations.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from only_llm_evaluator import EvaluationMetrics, ArticleResult


class ResultsVisualizer:
    """Handles visualization and comparison of evaluation results."""
    
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
        output_path = self.output_dir / "comparison_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {output_path}")
        
        return str(output_path)


class ComparisonReporter:
    """Handles comparison reporting between evaluation results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_and_report(
        self, 
        llama_results: List[ArticleResult],
        llama_metrics: EvaluationMetrics,
        rag_results: List[ArticleResult],
        rag_metrics: EvaluationMetrics
    ):
        """Generate comprehensive comparison report."""
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
            print(f"✓ RAG outperforms Llama by {improvement:.1f}% in accuracy")
        else:
            decline = (llama_metrics.accuracy - rag_metrics.accuracy) / llama_metrics.accuracy * 100
            print(f"✗ RAG underperforms Llama by {decline:.1f}% in accuracy")
        
        if rag_metrics.avg_time_per_article > llama_metrics.avg_time_per_article:
            time_increase = rag_metrics.avg_time_per_article - llama_metrics.avg_time_per_article
            print(f"⚠ RAG is {time_increase:.2f}s slower per article")
        else:
            time_decrease = llama_metrics.avg_time_per_article - rag_metrics.avg_time_per_article
            print(f"✓ RAG is {time_decrease:.2f}s faster per article")
        
        # Save detailed report
        self._save_comparison_report(llama_results, llama_metrics, rag_results, rag_metrics)
    
    def _save_comparison_report(
        self,
        llama_results: List[ArticleResult],
        llama_metrics: EvaluationMetrics,
        rag_results: List[ArticleResult],
        rag_metrics: EvaluationMetrics
    ):
        """Save detailed comparison report to JSON."""
        report_data = {
            "comparison_summary": {
                "llama_metrics": llama_metrics.to_dict(),
                "rag_metrics": rag_metrics.to_dict(),
                "improvements": {
                    "accuracy": rag_metrics.accuracy - llama_metrics.accuracy,
                    "precision": rag_metrics.precision - llama_metrics.precision,
                    "recall": rag_metrics.recall - llama_metrics.recall,
                    "f1_score": rag_metrics.f1_score - llama_metrics.f1_score,
                    "time_per_article": rag_metrics.avg_time_per_article - llama_metrics.avg_time_per_article
                }
            },
            "detailed_results": {
                "llama_results": [asdict(result) for result in llama_results],
                "rag_results": [asdict(result) for result in rag_results]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed comparison report saved to: {report_path}")
        
        return str(report_path)
