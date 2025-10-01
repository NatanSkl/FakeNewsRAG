"""
Reranking Analysis Module for Fake News RAG System

This module provides detailed analysis of different reranking mechanisms,
including cross-encoder reranking, MMR diversity, and xQuAD diversification.
"""

import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from rag_pipeline import load_store, RetrievalConfig, classify_article_rag, retrieve_evidence
from common.llm_client import LocalLLM, Llama
from evaluation_suite import TestArticle, EvaluationResult, FakeNewsEvaluator


@dataclass
class RerankingConfig:
    """Configuration for reranking analysis."""
    test_size: int = 40
    random_seed: int = 42
    temperature: float = 0.1
    max_tokens: int = 300
    save_detailed_results: bool = True
    output_dir: str = "reranking_analysis"


class RerankingAnalyzer:
    """Analyze the impact of different reranking mechanisms."""
    
    def __init__(self, store_dir: str = "mini_index/store", config: RerankingConfig = None):
        self.store_dir = store_dir
        self.config = config or RerankingConfig()
        self.store = load_store(store_dir)
        self.test_articles: List[TestArticle] = []
        self.results: List[EvaluationResult] = []
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def generate_test_dataset(self) -> List[TestArticle]:
        """Generate test dataset for reranking analysis."""
        print("Generating test dataset for reranking analysis...")
        
        # Sample diverse articles
        fake_chunks = [chunk for chunk in self.store.chunks if chunk.get("label") == "fake"]
        credible_chunks = [chunk for chunk in self.store.chunks if chunk.get("label") == "credible"]
        
        n_per_class = self.config.test_size // 2
        sampled_fake = random.sample(fake_chunks, min(n_per_class, len(fake_chunks)))
        sampled_credible = random.sample(credible_chunks, min(n_per_class, len(credible_chunks)))
        
        test_articles = []
        
        # Create test articles
        for i, chunk in enumerate(sampled_fake):
            test_articles.append(TestArticle(
                id=f"fake_{i}",
                title=chunk.get("title", "Untitled"),
                content=chunk.get("chunk_text", ""),
                true_label="fake",
                source_domain=chunk.get("source_domain", ""),
                published_date=chunk.get("published_date", "")
            ))
        
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
        print(f"Generated {len(test_articles)} test articles for reranking analysis")
        return test_articles
    
    def evaluate_no_reranking(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate without any reranking mechanisms."""
        print("Evaluating without reranking mechanisms...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=0,  # Disable MMR
            mmr_lambda=0.0
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
            method_name="RAG",
            config_name="no_reranking",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_mmr_only(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with only MMR diversity reranking."""
        print("Evaluating with MMR diversity reranking...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=False,
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
            method_name="RAG",
            config_name="mmr_only",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_cross_encoder_only(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with only cross-encoder reranking."""
        print("Evaluating with cross-encoder reranking...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=True,
            use_xquad=False,
            sent_maxpool=False,
            mmr_k=0,
            mmr_lambda=0.0
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
            method_name="RAG",
            config_name="cross_encoder_only",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_sentence_maxpool_only(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with only sentence maxpool reranking."""
        print("Evaluating with sentence maxpool reranking...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=False,
            use_xquad=False,
            sent_maxpool=True,
            mmr_k=0,
            mmr_lambda=0.0
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
            method_name="RAG",
            config_name="sentence_maxpool_only",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_xquad_only(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with only xQuAD diversification."""
        print("Evaluating with xQuAD diversification...")
        
        config = RetrievalConfig(
            topn=12,
            use_cross_encoder=False,
            use_xquad=True,
            sent_maxpool=False,
            mmr_k=0,
            mmr_lambda=0.0
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
            method_name="RAG",
            config_name="xquad_only",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def evaluate_combined_reranking(self, llm: LocalLLM) -> EvaluationResult:
        """Evaluate with combined reranking mechanisms."""
        print("Evaluating with combined reranking mechanisms...")
        
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
            method_name="RAG",
            config_name="combined_reranking",
            predictions=predictions,
            true_labels=[article.true_label for article in self.test_articles],
            confidence_scores=confidence_scores,
            processing_times=processing_times
        )
    
    def analyze_retrieval_quality(self, config_name: str) -> Dict[str, Any]:
        """Analyze retrieval quality for a specific configuration."""
        print(f"Analyzing retrieval quality for {config_name}...")
        
        # Get configuration
        configs = {
            "no_reranking": RetrievalConfig(topn=12, use_cross_encoder=False, use_xquad=False, 
                                          sent_maxpool=False, mmr_k=0, mmr_lambda=0.0),
            "mmr_only": RetrievalConfig(topn=12, use_cross_encoder=False, use_xquad=False, 
                                      sent_maxpool=False, mmr_k=60, mmr_lambda=0.4),
            "cross_encoder_only": RetrievalConfig(topn=12, use_cross_encoder=True, use_xquad=False, 
                                                  sent_maxpool=False, mmr_k=0, mmr_lambda=0.0),
            "sentence_maxpool_only": RetrievalConfig(topn=12, use_cross_encoder=False, use_xquad=False, 
                                                   sent_maxpool=True, mmr_k=0, mmr_lambda=0.0),
            "xquad_only": RetrievalConfig(topn=12, use_cross_encoder=False, use_xquad=True, 
                                         sent_maxpool=False, mmr_k=0, mmr_lambda=0.0),
            "combined_reranking": RetrievalConfig(topn=12, use_cross_encoder=True, use_xquad=True, 
                                                 sent_maxpool=True, mmr_k=60, mmr_lambda=0.4)
        }
        
        config = configs[config_name]
        
        retrieval_metrics = {
            'avg_retrieval_time': [],
            'avg_num_results': [],
            'avg_score_variance': [],
            'domain_diversity': []
        }
        
        for article in self.test_articles[:10]:  # Sample for analysis
            start_time = time.time()
            
            # Retrieve evidence for both labels
            fake_hits = retrieve_evidence(
                self.store, article.content, article.title,
                label_name="fake", cfg=config
            )
            credible_hits = retrieve_evidence(
                self.store, article.content, article.title,
                label_name="credible", cfg=config
            )
            
            retrieval_time = time.time() - start_time
            retrieval_metrics['avg_retrieval_time'].append(retrieval_time)
            retrieval_metrics['avg_num_results'].append(len(fake_hits) + len(credible_hits))
            
            # Analyze score variance
            all_scores = [h.get('_score', h.get('rrf', 0.0)) for h in fake_hits + credible_hits]
            if all_scores:
                retrieval_metrics['avg_score_variance'].append(np.var(all_scores))
            else:
                retrieval_metrics['avg_score_variance'].append(0.0)
            
            # Analyze domain diversity
            domains = set()
            for h in fake_hits + credible_hits:
                domain = h.get('source_domain', '')
                if domain:
                    domains.add(domain)
            retrieval_metrics['domain_diversity'].append(len(domains))
        
        # Calculate averages
        analysis_results = {
            'config_name': config_name,
            'avg_retrieval_time': np.mean(retrieval_metrics['avg_retrieval_time']),
            'avg_num_results': np.mean(retrieval_metrics['avg_num_results']),
            'avg_score_variance': np.mean(retrieval_metrics['avg_score_variance']),
            'avg_domain_diversity': np.mean(retrieval_metrics['domain_diversity']),
            'retrieval_efficiency': np.mean(retrieval_metrics['avg_num_results']) / np.mean(retrieval_metrics['avg_retrieval_time'])
        }
        
        return analysis_results
    
    def calculate_reranking_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate comprehensive reranking metrics."""
        reranking_metrics = {}
        
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
            fake_f1 = 2 * fake_precision * fake_recall / max(1, fake_precision + fake_recall)
            
            credible_correct = sum(1 for t, p in zip(y_true, y_pred) if t == "credible" and p == "credible")
            credible_total = sum(1 for t in y_true if t == "credible")
            credible_precision = credible_correct / max(1, sum(1 for p in y_pred if p == "credible"))
            credible_recall = credible_correct / max(1, credible_total)
            credible_f1 = 2 * credible_precision * credible_recall / max(1, credible_precision + credible_recall)
            
            # Weighted metrics
            weighted_precision = (fake_precision * fake_total + credible_precision * credible_total) / len(y_true)
            weighted_recall = (fake_recall * fake_total + credible_recall * credible_total) / len(y_true)
            weighted_f1 = 2 * weighted_precision * weighted_recall / max(1, weighted_precision + weighted_recall)
            
            # Timing metrics
            avg_time = np.mean(result.processing_times)
            total_time = np.sum(result.processing_times)
            
            # Confidence metrics
            avg_confidence = np.mean(result.confidence_scores)
            confidence_std = np.std(result.confidence_scores)
            
            reranking_metrics[result.config_name] = {
                'accuracy': accuracy,
                'precision_weighted': weighted_precision,
                'recall_weighted': weighted_recall,
                'f1_weighted': weighted_f1,
                'fake_precision': fake_precision,
                'fake_recall': fake_recall,
                'fake_f1': fake_f1,
                'credible_precision': credible_precision,
                'credible_recall': credible_recall,
                'credible_f1': credible_f1,
                'avg_processing_time': avg_time,
                'total_processing_time': total_time,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std
            }
        
        return reranking_metrics
    
    def run_reranking_analysis(self, llm: LocalLLM = None) -> Dict[str, Any]:
        """Run complete reranking analysis."""
        if llm is None:
            llm = Llama()
        
        print("Starting reranking analysis...")
        
        # Generate test dataset
        self.generate_test_dataset()
        
        # Run all reranking evaluations
        evaluations = [
            self.evaluate_no_reranking(llm),
            self.evaluate_mmr_only(llm),
            self.evaluate_cross_encoder_only(llm),
            self.evaluate_sentence_maxpool_only(llm),
            self.evaluate_xquad_only(llm),
            self.evaluate_combined_reranking(llm)
        ]
        
        self.results = evaluations
        
        # Calculate metrics
        metrics = self.calculate_reranking_metrics(evaluations)
        
        # Analyze retrieval quality
        retrieval_analysis = {}
        for config_name in ["no_reranking", "mmr_only", "cross_encoder_only", 
                           "sentence_maxpool_only", "xquad_only", "combined_reranking"]:
            retrieval_analysis[config_name] = self.analyze_retrieval_quality(config_name)
        
        # Generate summary
        summary = self._generate_reranking_summary(metrics, retrieval_analysis)
        
        if self.config.save_detailed_results:
            self._save_reranking_results(metrics, retrieval_analysis, summary)
        
        return {
            'results': evaluations,
            'metrics': metrics,
            'retrieval_analysis': retrieval_analysis,
            'summary': summary
        }
    
    def _generate_reranking_summary(self, metrics: Dict[str, Any], 
                                  retrieval_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reranking analysis summary."""
        # Find best performing configurations
        best_accuracy = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(metrics.items(), key=lambda x: x[1]['f1_weighted'])
        fastest = min(metrics.items(), key=lambda x: x[1]['avg_processing_time'])
        most_efficient = max(retrieval_analysis.items(), key=lambda x: x[1]['retrieval_efficiency'])
        
        # Calculate improvements over baseline
        baseline_accuracy = metrics.get('no_reranking', {}).get('accuracy', 0)
        improvements = {}
        
        for config, metric in metrics.items():
            if config != 'no_reranking':
                improvement = metric['accuracy'] - baseline_accuracy
                improvements[config] = {
                    'accuracy_gain': improvement,
                    'relative_improvement': improvement / baseline_accuracy if baseline_accuracy > 0 else 0
                }
        
        return {
            'best_accuracy': {
                'config': best_accuracy[0],
                'score': best_accuracy[1]['accuracy']
            },
            'best_f1': {
                'config': best_f1[0],
                'score': best_f1[1]['f1_weighted']
            },
            'fastest': {
                'config': fastest[0],
                'time': fastest[1]['avg_processing_time']
            },
            'most_efficient': {
                'config': most_efficient[0],
                'efficiency': most_efficient[1]['retrieval_efficiency']
            },
            'improvements': improvements,
            'total_configurations': len(metrics)
        }
    
    def _save_reranking_results(self, metrics: Dict[str, Any], 
                              retrieval_analysis: Dict[str, Any], 
                              summary: Dict[str, Any]):
        """Save reranking analysis results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_dir / "reranking_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save retrieval analysis
        with open(output_dir / "retrieval_analysis.json", "w") as f:
            json.dump(retrieval_analysis, f, indent=2)
        
        # Save summary
        with open(output_dir / "reranking_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create comparison table
        comparison_data = []
        for config, metric in metrics.items():
            retrieval_info = retrieval_analysis.get(config, {})
            comparison_data.append({
                'Configuration': config,
                'Accuracy': f"{metric['accuracy']:.3f}",
                'F1-Score': f"{metric['f1_weighted']:.3f}",
                'Avg Time (s)': f"{metric['avg_processing_time']:.2f}",
                'Retrieval Time (s)': f"{retrieval_info.get('avg_retrieval_time', 0):.2f}",
                'Domain Diversity': f"{retrieval_info.get('avg_domain_diversity', 0):.1f}",
                'Efficiency': f"{retrieval_info.get('retrieval_efficiency', 0):.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_dir / "reranking_comparison_table.csv", index=False)
        
        print(f"Reranking analysis results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    analyzer = RerankingAnalyzer()
    
    # Run analysis
    results = analyzer.run_reranking_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("RERANKING ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results['summary']
    print(f"Best Accuracy: {summary['best_accuracy']['config']} ({summary['best_accuracy']['score']:.3f})")
    print(f"Best F1-Score: {summary['best_f1']['config']} ({summary['best_f1']['score']:.3f})")
    print(f"Fastest: {summary['fastest']['config']} ({summary['fastest']['time']:.2f}s)")
    print(f"Most Efficient: {summary['most_efficient']['config']} ({summary['most_efficient']['efficiency']:.2f})")
    
    print("\nImprovements over baseline:")
    for config, improvement in summary['improvements'].items():
        print(f"  {config}: +{improvement['accuracy_gain']:.3f} ({improvement['relative_improvement']:.1%})")