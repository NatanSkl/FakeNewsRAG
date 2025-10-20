"""
RAG Evaluator

Evaluates RAG pipeline for fake news classification.
"""

import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG components
from pipeline.rag_pipeline import classify_article_rag, RAGOutput
from retrieval import load_store, RetrievalConfig
from common.llm_client import Llama

# Import shared classes
from only_llm_evaluator import EvaluationMetrics, ArticleResult


class RAGEvaluator:
    """Evaluator for RAG pipeline classification."""
    
    def __init__(self, store_path: str, output_dir: str = "evaluate/results"):
        self.store_path = store_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = None
        self.store = None
        
        print("="*80)
        print("RAG EVALUATION SYSTEM")
        print("="*80)
        print("Professional evaluation with real metrics")
        print("All output in English")
        print()
    
    def initialize(self, model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"):
        """Initialize the LLM model and vector store."""
        print("INITIALIZATION")
        print("-"*80)
        
        # Load Llama model
        print("Step 1/2: Loading Llama 3.2 3B model...")
        try:
            from llama_cpp import Llama as LlamaCpp
            self.llm = LlamaCpp(
                model_path=model_path,
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
    
    def run_rag_pipeline(self, articles: List[Dict], retrieval_config: RetrievalConfig = None) -> Tuple[List[ArticleResult], EvaluationMetrics]:
        """Run RAG pipeline evaluation on articles."""
        print("\nTEST: RAG PIPELINE")
        print("-"*80)
        print(f"Testing on {len(articles)} articles...")
        
        results = []
        start_total = time.time()
        
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(k=10, ce_model=None, diversity_type=None, verbose=False)
        
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
                    retrieval_config=retrieval_config,
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
    
    def save_results(self, results: List[ArticleResult], metrics: EvaluationMetrics, filename: str = "rag_results.json"):
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
