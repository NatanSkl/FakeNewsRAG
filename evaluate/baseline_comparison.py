"""
Baseline Comparison Module for Fake News RAG System

This module provides detailed comparison between the RAG pipeline and simple LLM baselines,
including various LLM configurations and prompt engineering approaches.
"""

import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

from rag_pipeline import load_store, RetrievalConfig, classify_article_rag
from classify.classifier import classify_article_simple
from common.llm_client import LocalLLM, Llama, Mistral
from evaluation_suite import TestArticle, EvaluationResult, FakeNewsEvaluator


@dataclass
class BaselineConfig:
    """Configuration for baseline comparisons."""
    test_size: int = 30
    random_seed: int = 42
    temperature: float = 0.1
    max_tokens: int = 300
    save_detailed_results: bool = True
    output_dir: str = "baseline_comparison"


class BaselineComparator:
    """Compare RAG pipeline against various baseline methods."""
    
    def __init__(self, store_dir: str = "mini_index/store", config: BaselineConfig = None):
        self.store_dir = store_dir
        self.config = config or BaselineConfig()
        self.store = load_store(store_dir)
        self.test_articles: List[TestArticle] = []
        self.results: List[EvaluationResult] = []
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def generate_test_dataset(self) -> List[TestArticle]:
        """Generate a focused test dataset for baseline comparison."""
        print("Generating test dataset for baseline comparison...")
        
        # Sample diverse articles from the index
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
        print(f"Generated {len(test_articles)} test articles for baseline comparison")
        return test_articles
    
    def evaluate_simple_prompt(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with simple classification prompt."""
        print("Evaluating Simple Prompt baseline...")
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            prompt = f"""
Classify this article as FAKE NEWS or CREDIBLE NEWS.

Title: {article.title}
Content: {article.content[:800]}

Respond with:
Classification: [FAKE/CREDIBLE]
Confidence: [0.0-1.0]
"""
            
            response = llm.simple(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            pred, conf = self._parse_response(response)
            predictions.append(pred)
            confidence_scores.append(conf)
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="Simple Prompt",
            config_name="baseline",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_structured_prompt(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with structured fact-checking prompt."""
        print("Evaluating Structured Prompt baseline...")
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            prompt = f"""
You are a fact-checker. Analyze this article for credibility.

Title: {article.title}
Content: {article.content[:800]}

Consider these factors:
1. Source reliability
2. Factual claims
3. Writing quality
4. Sensationalism

Classify as FAKE NEWS or CREDIBLE NEWS.

Respond with:
Classification: [FAKE/CREDIBLE]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation]
"""
            
            response = llm.simple(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            pred, conf = self._parse_response(response)
            predictions.append(pred)
            confidence_scores.append(conf)
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="Structured Prompt",
            config_name="baseline",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_rag_basic(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate RAG pipeline with basic configuration."""
        print("Evaluating RAG Basic...")
        
        config = RetrievalConfig(
            topn=8,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=40,
            mmr_lambda=0.4
        )
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            try:
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
                predictions.append("credible")
                confidence_scores.append(0.5)
            
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="RAG Basic",
            config_name="basic",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_rag_enhanced(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate RAG pipeline with enhanced configuration."""
        print("Evaluating RAG Enhanced...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=True,
            use_xquad=True,
            sent_maxpool=True,
            mmr_k=60,
            mmr_lambda=0.4
        )
        
        predictions = []
        confidence_scores = []
        processing_times = []
        
        for article in self.test_articles:
            start_time = time.time()
            
            try:
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
                predictions.append("credible")
                confidence_scores.append(0.5)
            
            processing_times.append(time.time() - start_time)
        
        return EvaluationResult(
            method_name="RAG Enhanced",
            config_name="enhanced",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def _parse_response(self, response: str) -> Tuple[str, float]:
        """Parse LLM response."""
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
    
    def calculate_comparison_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate comprehensive comparison metrics."""
        comparison_metrics = {}
        
        for result in results:
            y_true = result.true_labels
            y_pred = result.predictions
            
            # Basic metrics
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            # Per-class metrics
            fake_correct = sum(1 for t, p in zip(y_true, y_pred) if t == "fake" and p == "fake")
            fake_total = sum(1 for t in y_true if t == "fake")
            fake_precision = fake_correct / max(1, sum(1 for p in y_pred if p == "fake"))
            fake_recall = fake_correct / max(1, fake_total)
            
            credible_correct = sum(1 for t, p in zip(y_true, y_pred) if t == "credible" and p == "credible")
            credible_total = sum(1 for t in y_true if t == "credible")
            credible_precision = credible_correct / max(1, sum(1 for p in y_pred if p == "credible"))
            credible_recall = credible_correct / max(1, credible_total)
            
            # Timing metrics
            avg_time = np.mean(result.processing_times)
            total_time = np.sum(result.processing_times)
            
            # Confidence metrics
            avg_confidence = np.mean(result.confidence_scores)
            confidence_std = np.std(result.confidence_scores)
            
            comparison_metrics[f"{result.method_name}_{result.config_name}"] = {
                'accuracy': accuracy,
                'fake_precision': fake_precision,
                'fake_recall': fake_recall,
                'fake_f1': 2 * fake_precision * fake_recall / max(1, fake_precision + fake_recall),
                'credible_precision': credible_precision,
                'credible_recall': credible_recall,
                'credible_f1': 2 * credible_precision * credible_recall / max(1, credible_precision + credible_recall),
                'avg_processing_time': avg_time,
                'total_processing_time': total_time,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std
            }
        
        return comparison_metrics
    
    def run_baseline_comparison(self, llm: LocalLLM = None) -> Dict[str, Any]:
        """Run complete baseline comparison."""
        if llm is None:
            llm = Llama()
        
        print("Starting baseline comparison...")
        
        # Generate test dataset
        self.generate_test_dataset()
        
        # Run all evaluations
        evaluations = [
            self.evaluate_simple_prompt(llm),
            self.evaluate_structured_prompt(llm),
            self.evaluate_rag_basic(llm),
            self.evaluate_rag_enhanced(llm)
        ]
        
        self.results = evaluations
        
        # Calculate metrics
        metrics = self.calculate_comparison_metrics(evaluations)
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(metrics)
        
        if self.config.save_detailed_results:
            self._save_comparison_results(metrics, summary)
        
        return {
            'results': evaluations,
            'metrics': metrics,
            'summary': summary
        }
    
    def _generate_comparison_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary."""
        # Find best performing methods
        best_accuracy = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        best_fake_f1 = max(metrics.items(), key=lambda x: x[1]['fake_f1'])
        best_credible_f1 = max(metrics.items(), key=lambda x: x[1]['credible_f1'])
        fastest = min(metrics.items(), key=lambda x: x[1]['avg_processing_time'])
        
        # Calculate improvements
        simple_accuracy = metrics.get('Simple Prompt_baseline', {}).get('accuracy', 0)
        rag_enhanced_accuracy = metrics.get('RAG Enhanced_enhanced', {}).get('accuracy', 0)
        improvement = rag_enhanced_accuracy - simple_accuracy if simple_accuracy > 0 else 0
        
        return {
            'best_accuracy': {
                'method': best_accuracy[0],
                'score': best_accuracy[1]['accuracy']
            },
            'best_fake_detection': {
                'method': best_fake_f1[0],
                'f1_score': best_fake_f1[1]['fake_f1']
            },
            'best_credible_detection': {
                'method': best_credible_f1[0],
                'f1_score': best_credible_f1[1]['credible_f1']
            },
            'fastest': {
                'method': fastest[0],
                'time': fastest[1]['avg_processing_time']
            },
            'rag_improvement': {
                'accuracy_gain': improvement,
                'relative_improvement': improvement / simple_accuracy if simple_accuracy > 0 else 0
            },
            'total_articles': len(self.test_articles)
        }
    
    def _save_comparison_results(self, metrics: Dict[str, Any], summary: Dict[str, Any]):
        """Save comparison results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_dir / "baseline_comparison_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save summary
        with open(output_dir / "baseline_comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create comparison table
        comparison_data = []
        for method, metric in metrics.items():
            comparison_data.append({
                'Method': method,
                'Accuracy': f"{metric['accuracy']:.3f}",
                'Fake F1': f"{metric['fake_f1']:.3f}",
                'Credible F1': f"{metric['credible_f1']:.3f}",
                'Avg Time (s)': f"{metric['avg_processing_time']:.2f}",
                'Avg Confidence': f"{metric['avg_confidence']:.3f}"
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_dir / "baseline_comparison_table.csv", index=False)
        
        print(f"Baseline comparison results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    comparator = BaselineComparator()
    
    # Run comparison
    results = comparator.run_baseline_comparison()
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE COMPARISON SUMMARY")
    print("="*60)
    
    summary = results['summary']
    print(f"Best Accuracy: {summary['best_accuracy']['method']} ({summary['best_accuracy']['score']:.3f})")
    print(f"Best Fake Detection: {summary['best_fake_detection']['method']} ({summary['best_fake_detection']['f1_score']:.3f})")
    print(f"Best Credible Detection: {summary['best_credible_detection']['method']} ({summary['best_credible_detection']['f1_score']:.3f})")
    print(f"Fastest: {summary['fastest']['method']} ({summary['fastest']['time']:.2f}s)")
    print(f"RAG Improvement: {summary['rag_improvement']['accuracy_gain']:.3f} ({summary['rag_improvement']['relative_improvement']:.1%})")