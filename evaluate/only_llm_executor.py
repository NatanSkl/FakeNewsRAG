"""
Only LLM Executor

Executes LLM baseline classification on CSV files and saves results.
"""

import time
import pandas as pd
import tiktoken
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv('params.env')

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.evaluator import ArticleResult
from common.llm_client import Llama


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
class ClassificationResult:
    """Result of a single article classification."""
    article_id: str
    true_label: str
    predicted_label: str
    confidence: float
    processing_time: float
    correct: bool
    explanation: str


class OnlyLLMExecutor:
    """Executor for LLM-only classification that processes CSV files."""
    
    def __init__(self, llm_url: str = "http://127.0.0.1:8010"):
        self.llm_url = llm_url
        self.llm = None
        
        print("="*80)
        print("LLM EXECUTOR")
        print("="*80)
        print("Processes CSV files and executes LLM classification")
        print(f"Using LLM server at: {llm_url}")
        print()
    
    def initialize(self):
        """Initialize the LLM client and test connection."""
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
        
        # Test the connection with a simple request
        print("Testing LLM server connection...")
        try:
            test_messages = [{"role": "user", "content": "Say 'test'"}]
            test_response = self.llm.chat(messages=test_messages, max_tokens=5)
            print(f"Connection test successful: {test_response.text.strip()}")
        except Exception as e:
            print(f"ERROR: LLM server connection test failed: {e}")
            print("The server is not responding properly!")
            raise
        
        print("Initialization complete!")
        print("="*80)
    
    def run_llm_baseline(self, csv_path: str, output_dir: str = "experiments", limit: int = None) -> str:
        """
        Run LLM baseline on a CSV file and save results.
        
        Args:
            csv_path: Path to input CSV file (e.g., /StudentData/preprocessed/val_sampled.csv)
            output_dir: Directory to save results (default: experiments)
            limit: Maximum number of articles to process (for testing)
            
        Returns:
            Path to the saved results CSV file
        """
        print(f"\nRUNNING LLM BASELINE")
        print("-"*80)
        print(f"Input CSV: {csv_path}")
        
        # Load CSV data
        articles = self._load_csv_data(csv_path)
        print(f"Loaded {len(articles)} articles")
        
        # Apply limit if specified
        if limit:
            articles = articles[:limit]
            print(f"Limited to {len(articles)} articles for testing")
        
        # Execute classification
        results = self._classify_articles(articles)
        
        # Generate output filename
        input_filename = Path(csv_path).stem  # e.g., "val_sampled"
        output_filename = f"{input_filename}_llm_baseline.csv"
        
        # Save results
        output_path = self._save_results(results, output_dir, output_filename)
        
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
    
    def _classify_articles(self, articles: List[Dict]) -> List[ClassificationResult]:
        """Classify all articles using LLM."""
        print(f"Classifying {len(articles)} articles...")
        
        results = []
        
        for i, article in enumerate(articles, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(articles)} articles processed...")
            
            start_time = time.time()
            
            # Create prompt with token-limited content
            article_content_tokens = int(os.getenv('ARTICLE_CONTENT_TOKENS'))
            content_trimmed = _trim_tokens(article['content'], article_content_tokens)
            prompt = f"""Classify this news article as FAKE or RELIABLE and provide a brief explanation.

Title: {article['title']}
Content: {content_trimmed}

Please respond in the following format:
Classification: [FAKE/RELIABLE]
Explanation: [Brief explanation of your reasoning]"""
            
            # Use the Llama client's chat method
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(
                messages=messages,
                max_tokens=100  # Increased for explanation
            )
            
            response_text = response.text.strip()
            
            # Parse prediction and explanation
            prediction = None
            explanation = ""
            
            # Look for classification line
            lines = response_text.split('\n')
            for line in lines:
                line_upper = line.upper()
                if "CLASSIFICATION:" in line_upper:
                    if "FAKE" in line_upper:
                        prediction = "fake"
                    elif "RELIABLE" in line_upper or "CREDIBLE" in line_upper or "REAL" in line_upper:
                        prediction = "reliable"
                elif "EXPLANATION:" in line_upper:
                    explanation = line.split(":", 1)[1].strip() if ":" in line else ""
            
            # If no structured response, try to extract from text
            if prediction is None:
                response_upper = response_text.upper()
                if "FAKE" in response_upper:
                    prediction = "fake"
                elif "RELIABLE" in response_upper or "CREDIBLE" in response_upper or "REAL" in response_upper:
                    prediction = "reliable"
                else:
                    print(f"ERROR: Unclear LLM response for article {i}: '{response_text}'")
                    print(f"Expected 'FAKE' or 'RELIABLE', but got: '{response_text}'")
                    raise ValueError(f"LLM returned unclear response: '{response_text}'")
            
            # If no explanation found, use the full response as explanation
            if not explanation:
                explanation = response_text
            
            confidence = 0.8  # Llama doesn't provide confidence
            
            elapsed = time.time() - start_time
            
            result = ClassificationResult(
                article_id=article.get('id', f'article_{i}'),
                true_label=article['label'],
                predicted_label=prediction,
                confidence=confidence,
                processing_time=elapsed,
                correct=(prediction == article['label']),
                explanation=explanation
            )
            results.append(result)
        
        return results
    
    def _save_results(self, results: List[ClassificationResult], output_dir: str, filename: str) -> str:
        """Save classification results to CSV file."""
        # Create output directory
        output_path_obj = Path(output_dir)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        
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
                'explanation': result.explanation
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = output_path_obj / filename
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        total_articles = len(results)
        correct_predictions = sum(1 for r in results if r.correct)
        accuracy = correct_predictions / total_articles if total_articles > 0 else 0
        avg_time = sum(r.processing_time for r in results) / total_articles if total_articles > 0 else 0
        
        print(f"\nClassification Summary:")
        print(f"  Total articles: {total_articles}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Average processing time: {avg_time:.2f}s per article")
        
        return str(output_path)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM baseline on CSV file")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="experiments", help="Output directory")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8010", help="LLM server URL")
    parser.add_argument("--limit", type=int, help="Limit number of articles to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize executor
    executor = OnlyLLMExecutor(args.llm_url)
    executor.initialize()
    
    # Run baseline
    output_path = executor.run_llm_baseline(args.csv_path, args.output_dir, args.limit)
    
    print(f"\nExecution complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
