"""
Evaluation Runner for Fake News RAG System
Professional evaluation comparing RAG pipeline with Llama baseline
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    batch_size: int = 500
    num_batches: int = 10
    test_data_path: str = "data/test.csv"
    output_dir: str = "evaluate/results"
    random_seed: int = 42
    
    # RAG configuration variants to test
    rag_topn_values: List[int] = None
    rag_use_cross_encoder: List[bool] = None
    rag_use_xquad: List[bool] = None
    
    # LLM settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 300
    
    def __post_init__(self):
        if self.rag_topn_values is None:
            self.rag_topn_values = [8, 12, 16]
        if self.rag_use_cross_encoder is None:
            self.rag_use_cross_encoder = [False, True]
        if self.rag_use_xquad is None:
            self.rag_use_xquad = [False, True]


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    config_name: str
    method: str
    predictions: List[str]
    true_labels: List[str]
    confidence_scores: List[float]
    processing_times: List[float]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_processing_time: float


class EvaluationRunner:
    """Run systematic evaluation of RAG vs Llama."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results: List[EvaluationResult] = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        np.random.seed(self.config.random_seed)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load and prepare test data in batches."""
        print(f"Loading test data from: {self.config.test_data_path}")
        
        if not Path(self.config.test_data_path).exists():
            raise FileNotFoundError(f"Test data not found: {self.config.test_data_path}")
        
        df = pd.read_csv(self.config.test_data_path)
        print(f"Loaded {len(df)} test samples")
        
        # Sample batches
        total_samples = min(
            self.config.batch_size * self.config.num_batches,
            len(df)
        )
        
        df_sampled = df.sample(n=total_samples, random_state=self.config.random_seed)
        
        print(f"Selected {len(df_sampled)} samples for evaluation")
        print(f"Label distribution: {df_sampled['label'].value_counts().to_dict()}")
        
        return df_sampled
    
    def create_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into manageable batches."""
        batches = []
        
        for i in range(0, len(df), self.config.batch_size):
            batch = df.iloc[i:i + self.config.batch_size]
            batches.append(batch)
            print(f"Batch {len(batches)}: {len(batch)} samples")
        
        return batches
    
    def evaluate_llama_baseline(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate Llama baseline on a batch."""
        from llama_cpp import Llama
        
        print("Evaluating Llama baseline...")
        
        # Load Llama model
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False
        )
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for idx, row in batch.iterrows():
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))[:600]
            
            prompt = f"""Classify the following article as FAKE or RELIABLE.

Title: {title}
Content: {content}

Classification:"""
            
            start_time = time.time()
            
            response = llm(
                prompt,
                max_tokens=20,
                temperature=self.config.llm_temperature,
                stop=["\n"]
            )
            
            processing_times.append(time.time() - start_time)
            
            response_text = response['choices'][0]['text'].strip().upper()
            
            if "FAKE" in response_text:
                pred = "fake"
                conf = 0.75
            elif "RELIABLE" in response_text or "CREDIBLE" in response_text:
                pred = "reliable"
                conf = 0.75
            else:
                pred = "reliable"
                conf = 0.5
            
            predictions.append(pred)
            confidence_scores.append(conf)
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'processing_times': processing_times
        }
    
    def evaluate_rag_pipeline(self, batch: pd.DataFrame, 
                             topn: int = 12,
                             use_cross_encoder: bool = True,
                             use_xquad: bool = False) -> Dict[str, Any]:
        """Evaluate RAG pipeline on a batch."""
        from pipeline.rag_pipeline import classify_article_rag
        from llama_cpp import Llama
        
        config_name = f"RAG_topn{topn}_ce{use_cross_encoder}_xq{use_xquad}"
        print(f"Evaluating {config_name}...")
        
        # Load Llama for RAG
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False
        )
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for idx, row in batch.iterrows():
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            
            start_time = time.time()
            
            try:
                result = classify_article_rag(
                    article_title=title,
                    article_content=content,
                    llm=llm,
                    topn_per_label=topn
                )
                
                pred = result.classification.prediction
                conf = result.classification.confidence
                
            except Exception as e:
                print(f"Error in RAG classification: {e}")
                pred = "reliable"
                conf = 0.5
            
            processing_times.append(time.time() - start_time)
            predictions.append(pred)
            confidence_scores.append(conf)
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'processing_times': processing_times
        }
    
    def calculate_metrics(self, predictions: List[str], 
                         true_labels: List[str],
                         confidence_scores: List[float],
                         processing_times: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        
        # Convert labels to binary
        y_true = [1 if label == 'fake' else 0 for label in true_labels]
        y_pred = [1 if pred == 'fake' else 0 for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': np.sum(processing_times)
        }
    
    def run_evaluation(self):
        """Run complete evaluation."""
        print("="*60)
        print("FAKE NEWS RAG EVALUATION SYSTEM")
        print("="*60)
        
        # Load data
        df = self.load_test_data()
        batches = self.create_batches(df)
        
        # Run on first batch for testing
        test_batch = batches[0]
        print(f"\nRunning evaluation on batch 1 ({len(test_batch)} samples)...")
        
        true_labels = test_batch['label'].tolist()
        
        # Test 1: Llama baseline
        print("\n1. Testing Llama baseline...")
        llama_results = self.evaluate_llama_baseline(test_batch)
        llama_metrics = self.calculate_metrics(
            llama_results['predictions'],
            true_labels,
            llama_results['confidence_scores'],
            llama_results['processing_times']
        )
        
        print(f"   Accuracy: {llama_metrics['accuracy']:.3f}")
        print(f"   F1-Score: {llama_metrics['f1_score']:.3f}")
        print(f"   Avg Time: {llama_metrics['avg_processing_time']:.2f}s")
        
        # Test 2: RAG pipeline
        print("\n2. Testing RAG pipeline...")
        rag_results = self.evaluate_rag_pipeline(
            test_batch,
            topn=12,
            use_cross_encoder=True,
            use_xquad=False
        )
        rag_metrics = self.calculate_metrics(
            rag_results['predictions'],
            true_labels,
            rag_results['confidence_scores'],
            rag_results['processing_times']
        )
        
        print(f"   Accuracy: {rag_metrics['accuracy']:.3f}")
        print(f"   F1-Score: {rag_metrics['f1_score']:.3f}")
        print(f"   Avg Time: {rag_metrics['avg_processing_time']:.2f}s")
        
        # Save results
        self.save_results(llama_metrics, rag_metrics)
        self.create_visualizations(llama_metrics, rag_metrics)
        
        return llama_metrics, rag_metrics
    
    def save_results(self, llama_metrics: Dict, rag_metrics: Dict):
        """Save evaluation results."""
        results = {
            'llama_baseline': llama_metrics,
            'rag_pipeline': rag_metrics,
            'improvement': {
                'accuracy': rag_metrics['accuracy'] - llama_metrics['accuracy'],
                'f1_score': rag_metrics['f1_score'] - llama_metrics['f1_score']
            }
        }
        
        output_file = Path(self.config.output_dir) / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def create_visualizations(self, llama_metrics: Dict, rag_metrics: Dict):
        """Create comparison visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = ['Llama Baseline', 'RAG Pipeline']
        
        # Accuracy comparison
        accuracies = [llama_metrics['accuracy'], rag_metrics['accuracy']]
        ax1.bar(methods, accuracies, color=['#3498db', '#e74c3c'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # F1-Score comparison
        f1_scores = [llama_metrics['f1_score'], rag_metrics['f1_score']]
        ax2.bar(methods, f1_scores, color=['#3498db', '#e74c3c'])
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score Comparison')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Processing time comparison
        times = [llama_metrics['avg_processing_time'], rag_metrics['avg_processing_time']]
        ax3.bar(methods, times, color=['#3498db', '#e74c3c'])
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Average Processing Time')
        for i, v in enumerate(times):
            ax3.text(i, v + 0.1, f'{v:.2f}s', ha='center')
        
        # Metrics summary
        metrics_data = {
            'Accuracy': accuracies,
            'Precision': [llama_metrics['precision'], rag_metrics['precision']],
            'Recall': [llama_metrics['recall'], rag_metrics['recall']],
            'F1-Score': f1_scores
        }
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax4.bar(x + i*width, values, width, label=metric)
        
        ax4.set_ylabel('Score')
        ax4.set_title('Overall Metrics Comparison')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(methods)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        output_file = Path(self.config.output_dir) / "comparison_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_file}")


def main():
    """Main evaluation function."""
    config = EvaluationConfig(
        batch_size=500,
        num_batches=10,
        test_data_path="data/test.csv"
    )
    
    runner = EvaluationRunner(config)
    
    try:
        llama_metrics, rag_metrics = runner.run_evaluation()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        
        print(f"\nLlama Baseline:")
        print(f"  Accuracy: {llama_metrics['accuracy']:.3f}")
        print(f"  F1-Score: {llama_metrics['f1_score']:.3f}")
        
        print(f"\nRAG Pipeline:")
        print(f"  Accuracy: {rag_metrics['accuracy']:.3f}")
        print(f"  F1-Score: {rag_metrics['f1_score']:.3f}")
        
        improvement = rag_metrics['accuracy'] - llama_metrics['accuracy']
        print(f"\nImprovement: {improvement:+.3f} ({improvement/llama_metrics['accuracy']:+.1%})")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
