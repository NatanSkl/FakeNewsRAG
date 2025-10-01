"""
Visualization Module for Fake News RAG System Evaluation

This module provides comprehensive visualization capabilities for evaluation results,
including performance comparisons, confusion matrices, and interactive plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    # Output settings
    save_plots: bool = True
    output_dir: str = "evaluation_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # Plot settings
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 12
    title_size: int = 16
    
    # Color settings
    color_palette: str = "husl"
    fake_color: str = "#e74c3c"
    credible_color: str = "#3498db"
    
    # Interactive settings
    create_interactive: bool = True


class EvaluationVisualizer:
    """Create comprehensive visualizations for evaluation results."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Set matplotlib parameters
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['figure.figsize'] = self.config.figure_size
        
        # Create output directory
        if self.config.save_plots:
            Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def create_performance_comparison(self, metrics: Dict[str, Dict[str, float]], 
                                   output_file: str = "performance_comparison") -> None:
        """Create performance comparison visualizations."""
        print("Creating performance comparison plots...")
        
        # Prepare data
        methods = list(metrics.keys())
        accuracy_scores = [metrics[method]['accuracy'] for method in methods]
        f1_scores = [metrics[method]['f1_weighted'] for method in methods]
        processing_times = [metrics[method]['avg_processing_time'] for method in methods]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        bars1 = ax1.bar(methods, accuracy_scores, color=self.config.fake_color, alpha=0.7)
        ax1.set_title('Accuracy Comparison', fontsize=self.config.title_size)
        ax1.set_ylabel('Accuracy')
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars1, accuracy_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score comparison
        bars2 = ax2.bar(methods, f1_scores, color=self.config.credible_color, alpha=0.7)
        ax2.set_title('F1-Score Comparison', fontsize=self.config.title_size)
        ax2.set_ylabel('F1-Score')
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Processing time comparison
        bars3 = ax3.bar(methods, processing_times, color='orange', alpha=0.7)
        ax3.set_title('Average Processing Time', fontsize=self.config.title_size)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars3, processing_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 4. Accuracy vs Time scatter plot
        ax4.scatter(processing_times, accuracy_scores, s=100, alpha=0.7, 
                   c=range(len(methods)), cmap='viridis')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Processing Time', fontsize=self.config.title_size)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax4.annotate(method, (processing_times[i], accuracy_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrices(self, results: List[Any], output_file: str = "confusion_matrices") -> None:
        """Create confusion matrices for all methods."""
        print("Creating confusion matrices...")
        
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            # Calculate confusion matrix
            cm = confusion_matrix(result.true_labels, result.predictions, 
                               labels=['fake', 'credible'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Fake', 'Credible'],
                       yticklabels=['Fake', 'Credible'],
                       ax=axes[i])
            
            axes[i].set_title(f'{result.method_name}\n{result.config_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_per_class_metrics(self, metrics: Dict[str, Dict[str, float]], 
                               output_file: str = "per_class_metrics") -> None:
        """Create per-class metrics visualization."""
        print("Creating per-class metrics plots...")
        
        methods = list(metrics.keys())
        
        # Extract per-class metrics
        fake_precision = [metrics[method]['fake_precision'] for method in methods]
        fake_recall = [metrics[method]['fake_recall'] for method in methods]
        fake_f1 = [metrics[method]['fake_f1'] for method in methods]
        
        credible_precision = [metrics[method]['credible_precision'] for method in methods]
        credible_recall = [metrics[method]['credible_recall'] for method in methods]
        credible_f1 = [metrics[method]['credible_f1'] for method in methods]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Fake news metrics
        x = np.arange(len(methods))
        width = 0.25
        
        ax1.bar(x - width, fake_precision, width, label='Precision', alpha=0.8)
        ax1.bar(x, fake_recall, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, fake_f1, width, label='F1-Score', alpha=0.8)
        
        ax1.set_title('Fake News Detection Metrics', fontsize=self.config.title_size)
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Credible news metrics
        ax2.bar(x - width, credible_precision, width, label='Precision', alpha=0.8)
        ax2.bar(x, credible_recall, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, credible_f1, width, label='F1-Score', alpha=0.8)
        
        ax2.set_title('Credible News Detection Metrics', fontsize=self.config.title_size)
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Precision comparison
        ax3.bar(x - width/2, fake_precision, width, label='Fake', alpha=0.8)
        ax3.bar(x + width/2, credible_precision, width, label='Credible', alpha=0.8)
        ax3.set_title('Precision Comparison', fontsize=self.config.title_size)
        ax3.set_ylabel('Precision')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Recall comparison
        ax4.bar(x - width/2, fake_recall, width, label='Fake', alpha=0.8)
        ax4.bar(x + width/2, credible_recall, width, label='Credible', alpha=0.8)
        ax4.set_title('Recall Comparison', fontsize=self.config.title_size)
        ax4.set_ylabel('Recall')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_confidence_distribution(self, results: List[Any], 
                                     output_file: str = "confidence_distribution") -> None:
        """Create confidence score distribution plots."""
        print("Creating confidence distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, result in enumerate(results):
            if i >= len(axes):
                break
                
            # Separate confidence scores by prediction correctness
            correct_confidences = []
            incorrect_confidences = []
            
            for pred, true_label, conf in zip(result.predictions, result.true_labels, 
                                            result.confidence_scores):
                if pred == true_label:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
            
            # Create histogram
            ax = axes[i]
            ax.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', 
                   color=self.config.credible_color)
            ax.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', 
                   color=self.config.fake_color)
            
            ax.set_title(f'{result.method_name} - Confidence Distribution')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_xlim(0, 1)
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_processing_time_analysis(self, results: List[Any], 
                                      output_file: str = "processing_time_analysis") -> None:
        """Create processing time analysis plots."""
        print("Creating processing time analysis plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot of processing times
        processing_times_data = [result.processing_times for result in results]
        method_names = [f"{result.method_name}\n{result.config_name}" for result in results]
        
        ax1.boxplot(processing_times_data, labels=method_names)
        ax1.set_title('Processing Time Distribution')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cumulative processing time
        cumulative_times = [np.cumsum(times) for times in processing_times_data]
        
        for i, (cum_time, method) in enumerate(zip(cumulative_times, method_names)):
            ax2.plot(range(len(cum_time)), cum_time, label=method, linewidth=2)
        
        ax2.set_title('Cumulative Processing Time')
        ax2.set_xlabel('Article Index')
        ax2.set_ylabel('Cumulative Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, metrics: Dict[str, Dict[str, float]], 
                                   results: List[Any]) -> None:
        """Create interactive dashboard using Plotly."""
        if not self.config.create_interactive:
            return
        
        print("Creating interactive dashboard...")
        
        # Prepare data
        methods = list(metrics.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                          'Processing Time', 'Confidence vs Accuracy'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Accuracy comparison
        accuracy_scores = [metrics[method]['accuracy'] for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=accuracy_scores, name='Accuracy', 
                  marker_color=self.config.fake_color),
            row=1, col=1
        )
        
        # F1-Score comparison
        f1_scores = [metrics[method]['f1_weighted'] for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=f1_scores, name='F1-Score', 
                  marker_color=self.config.credible_color),
            row=1, col=2
        )
        
        # Processing time comparison
        processing_times = [metrics[method]['avg_processing_time'] for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=processing_times, name='Processing Time', 
                  marker_color='orange'),
            row=2, col=1
        )
        
        # Confidence vs Accuracy scatter
        avg_confidences = [metrics[method]['avg_confidence'] for method in methods]
        fig.add_trace(
            go.Scatter(x=avg_confidences, y=accuracy_scores, mode='markers+text',
                      text=methods, textposition='top center',
                      marker=dict(size=15, color=accuracy_scores, 
                                colorscale='Viridis', showscale=True),
                      name='Confidence vs Accuracy'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Fake News RAG System - Evaluation Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Methods", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Methods", row=1, col=2)
        fig.update_yaxes(title_text="F1-Score", row=1, col=2)
        fig.update_xaxes(title_text="Methods", row=2, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="Avg Confidence", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        # Save interactive plot
        if self.config.save_plots:
            output_dir = Path(self.config.output_dir)
            fig.write_html(str(output_dir / "interactive_dashboard.html"))
        
        print("Interactive dashboard created!")
    
    def create_reranking_comparison(self, reranking_results: Dict[str, Any], 
                                   output_file: str = "reranking_comparison") -> None:
        """Create reranking mechanism comparison plots."""
        print("Creating reranking comparison plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        configs = list(reranking_results.keys())
        accuracy_scores = [reranking_results[config]['accuracy'] for config in configs]
        f1_scores = [reranking_results[config]['f1_weighted'] for config in configs]
        processing_times = [reranking_results[config]['avg_processing_time'] for config in configs]
        recall_scores = [reranking_results[config]['recall_weighted'] for config in configs]
        
        # 1. Accuracy comparison
        bars1 = ax1.bar(configs, accuracy_scores, color=self.config.fake_color, alpha=0.7)
        ax1.set_title('Accuracy with Different Reranking Configurations')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, accuracy_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score comparison
        bars2 = ax2.bar(configs, f1_scores, color=self.config.credible_color, alpha=0.7)
        ax2.set_title('F1-Score with Different Reranking Configurations')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Processing time comparison
        bars3 = ax3.bar(configs, processing_times, color='orange', alpha=0.7)
        ax3.set_title('Processing Time with Different Reranking Configurations')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars3, processing_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 4. Accuracy vs Recall scatter
        ax4.scatter(recall_scores, accuracy_scores, s=100, alpha=0.7, c=range(len(configs)), cmap='viridis')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Recall Trade-off')
        
        # Add config labels
        for i, config in enumerate(configs):
            ax4.annotate(config, (recall_scores[i], accuracy_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.output_dir}/{output_file}.{self.config.plot_format}",
                       dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def create_all_visualizations(self, evaluation_results: Dict[str, Any]) -> None:
        """Create all visualizations from evaluation results."""
        print("Creating comprehensive visualizations...")
        
        metrics = evaluation_results.get('metrics', {})
        results = evaluation_results.get('results', [])
        
        if metrics:
            self.create_performance_comparison(metrics)
            self.create_per_class_metrics(metrics)
        
        if results:
            self.create_confusion_matrices(results)
            self.create_confidence_distribution(results)
            self.create_processing_time_analysis(results)
        
        if metrics and results:
            self.create_interactive_dashboard(metrics, results)
        
        print(f"All visualizations saved to {self.config.output_dir}")


if __name__ == "__main__":
    # Example usage
    visualizer = EvaluationVisualizer()
    
    # Example data structure
    example_metrics = {
        'Simple_LLM_baseline': {
            'accuracy': 0.75,
            'f1_weighted': 0.73,
            'fake_precision': 0.70,
            'fake_recall': 0.80,
            'fake_f1': 0.75,
            'credible_precision': 0.80,
            'credible_recall': 0.70,
            'credible_f1': 0.75,
            'avg_processing_time': 2.5,
            'avg_confidence': 0.78
        },
        'RAG_Enhanced_enhanced': {
            'accuracy': 0.85,
            'f1_weighted': 0.83,
            'fake_precision': 0.82,
            'fake_recall': 0.88,
            'fake_f1': 0.85,
            'credible_precision': 0.88,
            'credible_recall': 0.82,
            'credible_f1': 0.85,
            'avg_processing_time': 8.2,
            'avg_confidence': 0.82
        }
    }
    
    # Create visualizations
    visualizer.create_performance_comparison(example_metrics)
    print("Example visualizations created!")