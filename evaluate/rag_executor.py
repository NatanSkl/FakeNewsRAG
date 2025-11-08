"""
RAG Executor

Executes RAG pipeline classification on CSV files and saves results.
"""

import time
import pandas as pd
import tiktoken
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv('params.env')

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.evaluator import ArticleResult
from common.llm_client import Llama
from pipeline.rag_pipeline import classify_article_rag, RAGOutput
from retrieval.retrieval_v3 import load_store, RetrievalConfig
from generate.summary import Article, EvidenceChunk, contrastive_summaries
from classify.classifier import ClassificationResult


def _trim_tokens(s: str, max_tokens: int) -> str:
    """Trim text to approximately max_tokens using tiktoken encoding."""
    s = (s or "").strip()
    if not s:
        return s
    
    # Use cl100k_base encoding (GPT-4 tokenizer)
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(s)
    
    if len(tokens) <= max_tokens:
        return s
    
    # Truncate to max_tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)


@dataclass
class RAGClassificationResult:
    """Result of a single RAG article classification."""
    article_id: str
    true_label: str
    predicted_label: str
    confidence: float
    processing_time: float
    correct: bool
    reasoning: str
    fake_summary: str
    reliable_summary: str
    fake_evidence_count: int
    reliable_evidence_count: int
    retrieval_config: str
    # Debug fields
    fake_prompt: str = ""
    reliable_prompt: str = ""
    classification_prompt: str = ""


class RAGExecutor:
    """Executor for RAG pipeline classification that processes CSV files."""
    
    def __init__(self, llm_url: str = "http://127.0.0.1:8010", store_path: str = "/StudentData/index", debug_mode: bool = True):
        self.llm_url = llm_url
        self.store_path = store_path
        self.debug_mode = debug_mode
        self.llm = None
        self.stores = None
        
        print("="*80)
        print("RAG EXECUTOR")
        print("="*80)
        print("Processes CSV files and executes RAG pipeline classification")
        print(f"Using LLM server at: {llm_url}")
        print(f"Using store base path at: {store_path}")
        print()
    
    def initialize(self):
        """Initialize the LLM client, store, and test connections."""
        print("INITIALIZATION")
        print("-"*80)
        
        # Initialize Llama client
        print("Connecting to Llama server...")
        try:
            # Ensure base URL includes /v1
            base_url = self.llm_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
            
            self.llm = Llama(base_url=base_url)
            print("Llama client initialized successfully")
        except Exception as e:
            print(f"ERROR connecting to Llama server: {e}")
            print("Make sure the Llama server is running!")
            raise
        
        # Test the LLM connection
        print("Testing LLM server connection...")
        try:
            test_messages = [{"role": "user", "content": "Say 'test'"}]
            test_response = self.llm.chat(messages=test_messages, max_tokens=5)
            print(f"Connection test successful: {test_response.text.strip()}")
        except Exception as e:
            print(f"ERROR: LLM server connection test failed: {e}")
            print("The server is not responding properly!")
            raise
        
        # Load stores
        print("Loading stores...")
        try:
            fake_store = load_store(self.store_path + "_fake", verbose=True)
            reliable_store = load_store(self.store_path + "_reliable", verbose=True)
            self.stores = {"fake": fake_store, "reliable": reliable_store}
            print("Stores loaded successfully")
        except Exception as e:
            print(f"ERROR loading stores: {e}")
            print(f"Make sure the stores exist at: {self.store_path}_fake and {self.store_path}_reliable")
            raise
        
        print("Initialization complete!")
        print("="*80)
    
    def run_rag_pipeline(self, csv_path: str, output_dir: str = "experiments", 
                        retrieval_config: RetrievalConfig = None, 
                        prompt_type: int = 0, 
                        naming_convention: str = "fake_reliable",
                        limit: int = None,
                        fix_missing: bool = False) -> str:
        """
        Run RAG pipeline on a CSV file and save results.
        
        Args:
            csv_path: Path to input CSV file (e.g., /StudentData/preprocessed/val_sampled.csv)
            output_dir: Directory to save results (default: experiments)
            retrieval_config: Configuration for retrieval (uses defaults if None)
            prompt_type: Type of prompt to use for summarization and classification
            naming_convention: Naming convention for classification labels
            limit: Maximum number of articles to process (for testing)
            fix_missing: If True, only process rows with missing results
            
        Returns:
            Path to the saved results CSV file
        """
        print(f"\nRUNNING RAG PIPELINE")
        print("-"*80)
        print(f"Input CSV: {csv_path}")
        print(f"Retrieval config: {retrieval_config}")
        print(f"Prompt type: {prompt_type}")
        print(f"Naming convention: {naming_convention}")
        
        # Load CSV data
        articles = self._load_csv_data(csv_path)
        print(f"Loaded {len(articles)} articles")
        
        # Apply limit if specified
        if limit:
            articles = articles[:limit]
            print(f"Limited to {len(articles)} articles for testing")
        
        # Generate output filename based on configuration
        output_filename = self._generate_output_filename(csv_path, retrieval_config, prompt_type, naming_convention, limit)
        output_path = Path(output_dir) / output_filename
        
        # Handle fix_missing mode
        if fix_missing:
            print("Fix-missing mode: Only processing rows with missing results")
            articles = self._filter_missing_articles(articles, output_path)
            if not articles:
                print("No missing articles found to process")
                return str(output_path)
            print(f"Found {len(articles)} articles with missing results")
        
        # Execute classification
        results = self._classify_articles_rag(articles, retrieval_config, prompt_type, naming_convention)
        
        # Save results
        self._save_results(results, output_path)
        
        print(f"\nResults saved to: {output_path}")
        return str(output_path)
    
    def _load_csv_data(self, csv_path: str) -> List[Dict]:
        """Load articles from CSV file."""
        print(f"Loading data from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
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
    
    def _generate_output_filename(self, csv_path: str, retrieval_config: RetrievalConfig, 
                                prompt_type: int, naming_convention: str, limit: int = None) -> str:
        """Generate descriptive output filename based on configuration."""
        # Extract base filename (e.g., "val_sampled" from "/path/to/val_sampled.csv")
        input_filename = Path(csv_path).stem
        
        # Start with base filename
        parts = [input_filename, "rag"]
        
        # Add cross-encoder model name if specified
        if retrieval_config.ce_model_name:
            ce_name = retrieval_config.ce_model_name.split('/')[-1]  # Get just the model name
            parts.append(f"ce={ce_name}")
        else:
            parts.append("ce=None")
        
        # Add diversity type
        diversity = retrieval_config.diversity_type if retrieval_config.diversity_type else "None"
        parts.append(f"diversity={diversity}")
        
        # Add prompt type
        parts.append(f"prompt={prompt_type}")
        
        # Add naming convention
        parts.append(f"naming={naming_convention}")
        
        # Add limit if specified
        if limit:
            parts.append(f"limit={limit}")
        
        # Join parts and add .csv extension
        return "_".join(parts) + ".csv"
    
    def _filter_missing_articles(self, articles: List[Dict], output_path: Path) -> List[Dict]:
        """Filter articles to only include those with missing results in the output file."""
        if not output_path.exists():
            print("Output file doesn't exist yet, processing all articles")
            return articles
        
        try:
            existing_df = pd.read_csv(output_path)
            existing_ids = set(existing_df['article_id'].astype(str))
            
            # Filter for articles that are missing or have <MISSING> values
            missing_articles = []
            for article in articles:
                article_id = str(article['id'])
                if article_id not in existing_ids:
                    missing_articles.append(article)
                else:
                    # Check if any critical fields are missing
                    row = existing_df[existing_df['article_id'] == article_id].iloc[0]
                    if (pd.isna(row['predicted_label']) or 
                        str(row['predicted_label']).strip() == '<MISSING>' or
                        pd.isna(row['reasoning']) or 
                        str(row['reasoning']).strip() == '<MISSING>'):
                        missing_articles.append(article)
            
            return missing_articles
        except Exception as e:
            print(f"Error filtering missing articles: {e}")
            print("Processing all articles instead")
            return articles
    
    def _classify_articles_rag(self, articles: List[Dict], retrieval_config: RetrievalConfig, 
                              prompt_type: int, naming_convention: str) -> List[RAGClassificationResult]:
        """Classify all articles using RAG pipeline."""
        print(f"Classifying {len(articles)} articles using RAG pipeline...")
        
        # Use default retrieval config if none provided
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(
                k=10,
                ce_model=None,
                diversity_type=None,
                verbose=False
            )
        
        results = []
        
        for i, article in enumerate(articles, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(articles)} articles processed...")
            
            start_time = time.time()
            
            try:
                # Create Article object
                article_obj = Article(
                    id=article['id'],
                    title=article['title'],
                    text=article['content']
                )
                
                # Run RAG pipeline
                rag_output = classify_article_rag(
                    article_title=article['title'],
                    article_content=article['content'],
                    stores=self.stores,
                    llm=self.llm,
                    retrieval_config=retrieval_config,
                    verbose=False,
                    prompt_type=prompt_type,
                    naming_convention=naming_convention,
                    debug_mode=self.debug_mode
                )
                
                # Extract results
                classification = rag_output.classification
                fake_summary = rag_output.fake_summary
                reliable_summary = rag_output.reliable_summary
                fake_evidence_count = len(rag_output.fake_evidence)
                reliable_evidence_count = len(rag_output.reliable_evidence)
                
                # Extract debug information if available
                fake_prompt = ""
                reliable_prompt = ""
                classification_prompt = ""
                
                if self.debug_mode:
                    # Get debug info from summaries (if available)
                    summaries_debug = contrastive_summaries(
                        self.llm, article_obj, rag_output.fake_evidence, rag_output.reliable_evidence, 
                        promt_type=prompt_type, return_debug=True
                    )
                    fake_prompt = summaries_debug.get("fake_prompt", "")
                    reliable_prompt = summaries_debug.get("reliable_prompt", "")
                    classification_prompt = classification.classification_prompt
                
                elapsed = time.time() - start_time
                
                result = RAGClassificationResult(
                    article_id=article['id'],
                    true_label=article['label'],
                    predicted_label=classification.prediction,
                    confidence=classification.confidence,
                    processing_time=elapsed,
                    correct=(classification.prediction == article['label']),
                    reasoning=classification.reasoning,
                    fake_summary=fake_summary,
                    reliable_summary=reliable_summary,
                    fake_evidence_count=fake_evidence_count,
                    reliable_evidence_count=reliable_evidence_count,
                    retrieval_config=str(retrieval_config),
                    fake_prompt=fake_prompt,
                    reliable_prompt=reliable_prompt,
                    classification_prompt=classification_prompt
                )
                results.append(result)
                
            except Exception as e:
                print(f"ERROR processing article {i} (ID: {article['id']}): {e}")
                
                # Create result with missing values
                elapsed = time.time() - start_time
                result = RAGClassificationResult(
                    article_id=article['id'],
                    true_label=article['label'],
                    predicted_label="<MISSING>",
                    confidence=0.0,
                    processing_time=elapsed,
                    correct=False,
                    reasoning="<MISSING>",
                    fake_summary="<MISSING>",
                    reliable_summary="<MISSING>",
                    fake_evidence_count=0,
                    reliable_evidence_count=0,
                    retrieval_config=str(retrieval_config)
                )
                results.append(result)
        
        return results
    
    def _save_results(self, results: List[RAGClassificationResult], output_path: Path) -> str:
        """Save classification results to CSV file."""
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        data = []
        for result in results:
            data.append({
                'article_id': result.article_id,
                'true_label': result.true_label,
                'predicted_label': result.predicted_label,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'correct': result.correct,
                'reasoning': result.reasoning,
                'fake_summary': result.fake_summary,
                'reliable_summary': result.reliable_summary,
                'fake_evidence_count': result.fake_evidence_count,
                'reliable_evidence_count': result.reliable_evidence_count,
                'retrieval_config': result.retrieval_config,
                'fake_prompt': result.fake_prompt,
                'reliable_prompt': result.reliable_prompt,
                'classification_prompt': result.classification_prompt
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        total_articles = len(results)
        correct_predictions = sum(1 for r in results if r.correct)
        missing_predictions = sum(1 for r in results if r.predicted_label == "<MISSING>")
        accuracy = correct_predictions / (total_articles - missing_predictions) if (total_articles - missing_predictions) > 0 else 0
        avg_time = sum(r.processing_time for r in results) / total_articles if total_articles > 0 else 0
        
        print(f"\nClassification Summary:")
        print(f"  Total articles: {total_articles}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Missing predictions: {missing_predictions}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Average processing time: {avg_time:.2f}s per article")
        
        return str(output_path)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline on CSV file")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="experiments", help="Output directory")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8010", help="LLM server URL")
    parser.add_argument("--store-path", default="/StudentData/index", help="Path to store directory")
    parser.add_argument("--limit", type=int, help="Limit number of articles to process (for testing)")
    parser.add_argument("--fix-missing", action="store_true", help="Only process rows with missing results")
    
    # Retrieval config parameters
    parser.add_argument("--k", type=int, default=10, help="Number of evidence chunks to retrieve")
    parser.add_argument("--ce-model-name", help="Cross-encoder model name for reranking")
    parser.add_argument("--diversity-type", choices=["mmr"], help="Diversity method for retrieval")
    
    # Pipeline parameters
    parser.add_argument("--prompt-type", type=int, default=0, help="Prompt type for summarization and classification")
    parser.add_argument("--naming-convention", default="fake_reliable", help="Naming convention for classification labels")
    parser.add_argument("--debug-mode", action="store_true", default=True, help="Enable debug mode to save prompts")
    
    args = parser.parse_args()
    
    # Create retrieval config
    retrieval_config = RetrievalConfig(
        k=args.k,
        ce_model_name=args.ce_model_name,
        diversity_type=args.diversity_type,
        verbose=False
    )
    
    # Initialize executor
    executor = RAGExecutor(args.llm_url, args.store_path, debug_mode=args.debug_mode)
    executor.initialize()
    
    # Run RAG pipeline
    output_path = executor.run_rag_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        retrieval_config=retrieval_config,
        prompt_type=args.prompt_type,
        naming_convention=args.naming_convention,
        limit=args.limit,
        fix_missing=args.fix_missing
    )
    
    print(f"\nExecution complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
