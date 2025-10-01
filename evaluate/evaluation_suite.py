"""
Comprehensive Evaluation Suite for Fake News RAG System

This module provides a complete evaluation framework to test the RAG pipeline
against various baselines and configurations, including:
- RAG vs Simple LLM comparison
- Performance with/without reranking mechanisms
- Embedding analysis and comparison
- Classification metrics and visualizations
"""

import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from rag_pipeline import load_store, RetrievalConfig, classify_article_rag, RAGOutput
from classify.classifier import classify_article_simple
from common.llm_client import LocalLLM, Llama


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    # Dataset settings
    test_size: int = 50
    random_seed: int = 42
    
    # RAG configurations to test
    rag_configs: Dict[str, RetrievalConfig] = field(default_factory=dict)
    
    # LLM settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 300
    
    # Evaluation metrics
    top_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Output settings
    save_results: bool = True
    output_dir: str = "evaluation_results"
    create_plots: bool = True


@dataclass
class TestArticle:
    """Test article with ground truth label."""
    id: str
    title: str
    content: str
    true_label: str  # "fake" or "credible"
    source_domain: str = ""
    published_date: str = ""


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    method_name: str
    config_name: str
    predictions: List[str]
    true_labels: List[str]
    confidence_scores: List[float]
    processing_times: List[float]
    retrieval_metrics: Optional[Dict[str, float]] = None
    error_analysis: Optional[Dict[str, Any]] = None


class FakeNewsEvaluator:
    """Main evaluation class for the Fake News RAG system."""
    
    def __init__(self, store_dir: str = "mini_index/store", config: EvaluationConfig = None):
        self.store_dir = store_dir
        self.config = config or EvaluationConfig()
        self.store = load_store(store_dir)
        self.test_articles: List[TestArticle] = []
        self.results: List[EvaluationResult] = []
        
        # Initialize default RAG configurations
        self._setup_default_configs()
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def _setup_default_configs(self):
        """Setup default RAG configurations for testing."""
        self.config.rag_configs = {
            "rag_basic": RetrievalConfig(
                topn=12,
                use_cross_encoder=False,
                use_xquad=False,
                sent_maxpool=False,
                mmr_k=60,
                mmr_lambda=0.4
            ),
            "rag_with_reranking": RetrievalConfig(
                topn=12,
                use_cross_encoder=True,
                use_xquad=False,
                sent_maxpool=True,
                mmr_k=60,
                mmr_lambda=0.4
            ),
            "rag_full": RetrievalConfig(
                topn=12,
                use_cross_encoder=True,
                use_xquad=True,
                sent_maxpool=True,
                mmr_k=60,
                mmr_lambda=0.4
            )
        }
    
    def generate_test_dataset(self) -> List[TestArticle]:
        """Generate a test dataset from the existing index."""
        print("Generating test dataset from index...")
        
        # Sample articles from the index
        fake_chunks = [chunk for chunk in self.store.chunks if chunk.get("label") == "fake"]
        credible_chunks = [chunk for chunk in self.store.chunks if chunk.get("label") == "credible"]
        
        # Sample equal numbers from each class
        n_per_class = self.config.test_size // 2
        sampled_fake = random.sample(fake_chunks, min(n_per_class, len(fake_chunks)))
        sampled_credible = random.sample(credible_chunks, min(n_per_class, len(credible_chunks)))
        
        test_articles = []
        
        # Create test articles from fake chunks
        for i, chunk in enumerate(sampled_fake):
            test_articles.append(TestArticle(
                id=f"fake_{i}",
                title=chunk.get("title", "Untitled"),
                content=chunk.get("chunk_text", ""),
                true_label="fake",
                source_domain=chunk.get("source_domain", ""),
                published_date=chunk.get("published_date", "")
            ))
        
        # Create test articles from credible chunks
        for i, chunk in enumerate(sampled_credible):
            test_articles.append(TestArticle(
                id=f"credible_{i}",
                title=chunk.get("title", "Untitled"),
                content=chunk.get("chunk_text", ""),
                true_label="credible",
                source_domain=chunk.get("source_domain", ""),
                published_date=chunk.get("published_date", "")
            ))
        
        self.test_articles = test_articles
        print(f"Generated {len(test_articles)} test articles ({len(sampled_fake)} fake, {len(sampled_credible)} credible)")
        return test_articles
    
    def evaluate_simple_llm(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate simple LLM baseline (no RAG)."""
        print("Evaluating Simple LLM baseline...")
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            # Simple classification prompt
            prompt = f"""
Classify the following article as FAKE NEWS or CREDIBLE NEWS.

Title: {article.title}
Content: {article.content[:1000]}

Respond with:
Classification: [FAKE/CREDIBLE]
Confidence: [0.0-1.0]
"""
            
            response = llm.simple(
                prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            
            # Parse response
            pred, conf = self._parse_simple_response(response)
            predictions.append(pred)
            confidence_scores.append(conf)
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="Simple LLM",
            config_name="baseline",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_rag_pipeline(self, config_name: str, llm: LocalLLM) -> EvaluationResult:
        """Evaluate RAG pipeline with specific configuration."""
        print(f"Evaluating RAG pipeline with config: {config_name}")
        
        config = self.config.rag_configs[config_name]
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            try:
                # Run full RAG pipeline
                rag_output = classify_article_rag(
                    article_title=article.title,
                    article_content=article.content,
                    store_dir=self.store_dir,
                    llm=llm,
                    topn_per_label=config.topn
                )
                
                predictions.append(rag_output.classification.prediction)
                confidence_scores.append(rag_output.classification.confidence)
                
            except Exception as e:
                print(f"Error processing article {article.id}: {e}")
                predictions.append("credible")  # Default fallback
                confidence_scores.append(0.5)
            
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="RAG Pipeline",
            config_name=config_name,
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def _parse_simple_response(self, response: str) -> Tuple[str, float]:
        """Parse simple LLM response."""
        lines = response.strip().split('\n')
        prediction = "credible"  # default
        confidence = 0.5  # default
        
        for line in lines:
            line = line.strip()
            if line.startswith("Classification:"):
                classification_text = line.split(":", 1)[1].strip().upper()
                if "FAKE" in classification_text:
                    prediction = "fake"
                elif "CREDIBLE" in classification_text:
                    prediction = "credible"
            elif line.startswith("Confidence:"):
                try:
                    confidence_text = line.split(":", 1)[1].strip()
                    confidence = float(confidence_text)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5
        
        return prediction, confidence
    
    def calculate_metrics(self, result: EvaluationResult) -> Dict[str, float]:
        """Calculate comprehensive metrics for an evaluation result."""
        y_true = result.true_labels
        y_pred = result.predictions
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Confidence metrics
        avg_confidence = np.mean(result.confidence_scores)
        confidence_std = np.std(result.confidence_scores)
        
        # Timing metrics
        avg_processing_time = np.mean(result.processing_times)
        total_processing_time = np.sum(result.processing_times)
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_fake': precision_per_class[0] if len(precision_per_class) > 0 else 0,
            'recall_fake': recall_per_class[0] if len(recall_per_class) > 0 else 0,
            'f1_fake': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'precision_credible': precision_per_class[1] if len(precision_per_class) > 1 else 0,
            'recall_credible': recall_per_class[1] if len(recall_per_class) > 1 else 0,
            'f1_credible': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': total_processing_time
        }
    
    def run_complete_evaluation(self, llm: LocalLLM = None) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        if llm is None:
            llm = Llama()
        
        print("Starting complete evaluation suite...")
        
        # Generate test dataset
        self.generate_test_dataset()
        
        # Evaluate simple LLM baseline
        simple_result = self.evaluate_simple_llm(llm)
        self.results.append(simple_result)
        
        # Evaluate RAG configurations
        for config_name in self.config.rag_configs.keys():
            rag_result = self.evaluate_rag_pipeline(config_name, llm)
            self.results.append(rag_result)
        
        # Calculate metrics for all results
        all_metrics = {}
        for result in self.results:
            metrics = self.calculate_metrics(result)
            all_metrics[f"{result.method_name}_{result.config_name}"] = metrics
        
        # Generate summary
        summary = self._generate_evaluation_summary(all_metrics)
        
        if self.config.save_results:
            self._save_results(all_metrics, summary)
        
        return {
            'results': self.results,
            'metrics': all_metrics,
            'summary': summary
        }
    
    def _generate_evaluation_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate evaluation summary."""
        # Find best performing method
        best_accuracy = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(metrics.items(), key=lambda x: x[1]['f1_weighted'])
        fastest = min(metrics.items(), key=lambda x: x[1]['avg_processing_time'])
        
        return {
            'best_accuracy': {
                'method': best_accuracy[0],
                'score': best_accuracy[1]['accuracy']
            },
            'best_f1': {
                'method': best_f1[0],
                'score': best_f1[1]['f1_weighted']
            },
            'fastest': {
                'method': fastest[0],
                'time': fastest[1]['avg_processing_time']
            },
            'total_articles': len(self.test_articles),
            'total_configurations': len(self.results)
        }
    
    def _save_results(self, metrics: Dict[str, Dict[str, float]], summary: Dict[str, Any]):
        """Save evaluation results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save summary
        with open(output_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_data = []
        for result in self.results:
            for i, (pred, true_label, conf, time_taken) in enumerate(zip(
                result.predictions, result.true_labels, 
                result.confidence_scores, result.processing_times
            )):
                results_data.append({
                    'method': result.method_name,
                    'config': result.config_name,
                    'article_id': self.test_articles[i].id,
                    'prediction': pred,
                    'true_label': true_label,
                    'confidence': conf,
                    'processing_time': time_taken,
                    'correct': pred == true_label
                })
        
        df = pd.DataFrame(results_data)
        df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    evaluator = FakeNewsEvaluator()
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for method, metrics in results['metrics'].items():
        print(f"\n{method}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.3f}")
        print(f"  Avg Processing Time: {metrics['avg_processing_time']:.2f}s")