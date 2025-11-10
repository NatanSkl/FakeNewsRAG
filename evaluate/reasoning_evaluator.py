"""
Reasoning Evaluator

Evaluates how well the reasoning in classification results is supported by the article content.
Uses LLM to score reasoning support on a 0-1 scale.
"""

import time
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import re
import tiktoken
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('params.env')
STORAGE_DIR = os.getenv('STORAGE_DIR', '/StudentData/reproduce')

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

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


class ReasoningEvaluator:
    """Evaluates reasoning support from article content using LLM."""
    
    def __init__(self, llm_client: Optional[Llama] = None, articles_csv: str = None):
        """Initialize the reasoning evaluator.
        
        Args:
            llm_client: LLM client to use for evaluation. If None, creates a Llama client.
            articles_csv: Path to CSV file containing article content. If None, uses STORAGE_DIR/preprocessed/news_balanced.csv.
        """
        self.llm = llm_client or Llama()
        
        # Set default articles_csv path if not provided
        if articles_csv is None:
            articles_csv = str(Path(STORAGE_DIR) / "preprocessed" / "news_balanced.csv")
        
        print("="*80)
        print("REASONING SUPPORT EVALUATION SYSTEM")
        print("="*80)
        print("Evaluates how well reasoning is supported by article content")
        print("Scores reasoning support on a 0-1 scale using LLM")
        print()
        
        # Load article content
        print(f"Loading article content from: {articles_csv}")
        self.articles = self._load_articles(articles_csv)
        print(f"Loaded {len(self.articles)} articles")
        print()
    
    def _load_articles(self, csv_path: str) -> Dict[str, dict]:
        """Load articles from CSV file into a dictionary.
        
        Args:
            csv_path: Path to CSV file with article data
            
        Returns:
            Dictionary mapping article_id to article data (title, content, label)
        """
        df = pd.read_csv(csv_path)
        articles = {}
        
        for _, row in df.iterrows():
            article_id = str(row['id']).strip()
            # Remove .0 suffix if present (for float IDs)
            if article_id.endswith('.0'):
                article_id = article_id[:-2]
            
            articles[article_id] = {
                'title': str(row['title']),
                'content': str(row['content']),
                'label': str(row['label'])
            }
        
        return articles
    
    def evaluate_csv(self, csv_path: str) -> pd.DataFrame:
        """Evaluate reasoning support for all rows in a CSV file.
        
        Args:
            csv_path: Path to the CSV file to evaluate
            
        Returns:
            DataFrame with added reasoning_support_score column
        """
        csv_path = Path(csv_path)
        print(f"\nEVALUATING REASONING FROM: {csv_path}")
        print("-"*80)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Add new column for support scores
        support_scores = []
        skipped_count = 0
        
        # Process each row
        total_start_time = time.time()
        for idx, row in df.iterrows():
            row_start_time = time.time()
            print(f"\nProcessing row {idx + 1}/{len(df)}...")
            
            article_id = str(row.get('article_id', '')).strip()
            # Remove .0 suffix if present (for float IDs)
            if article_id.endswith('.0'):
                article_id = article_id[:-2]
            
            reasoning = str(row.get('reasoning', ''))
            predicted_label = str(row.get('predicted_label', ''))
            
            # Get article content
            if article_id not in self.articles:
                print(f"  WARNING: Article ID {article_id} not found in articles database!")
                print(f"  Skipping this row...")
                support_scores.append(None)
                skipped_count += 1
                continue
            
            article = self.articles[article_id]
            article_content = article['content']
            article_title = article['title']
            
            # Trim article content to configured token limit
            article_content_tokens = int(os.getenv('ARTICLE_CONTENT_TOKENS'))
            article_content_trimmed = _trim_tokens(article_content, article_content_tokens)
            
            # Evaluate support
            score = self._evaluate_reasoning_support(
                reasoning, predicted_label, article_title, article_content_trimmed
            )
            support_scores.append(score)
            
            row_elapsed = time.time() - row_start_time
            total_elapsed = time.time() - total_start_time
            avg_time = total_elapsed / (idx + 1)
            remaining_rows = len(df) - (idx + 1)
            estimated_remaining = avg_time * remaining_rows
            
            print(f"  Support score: {score:.3f}")
            print(f"  Time: {row_elapsed:.2f}s (avg: {avg_time:.2f}s/row, est. remaining: {estimated_remaining:.1f}s)")
        
        # Add scores to dataframe
        df['reasoning_support_score'] = support_scores
        
        # Calculate statistics (excluding None values)
        valid_scores = [s for s in support_scores if s is not None]
        total_time = time.time() - total_start_time
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        avg_time_per_row = total_time / len(df) if len(df) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        if skipped_count > 0:
            print(f"WARNING: Skipped {skipped_count} rows due to missing article IDs")
        print(f"Average support score: {avg_score:.3f}")
        print(f"Min score: {min(valid_scores):.3f}")
        print(f"Max score: {max(valid_scores):.3f}")
        print(f"Total time: {total_time:.1f}s ({avg_time_per_row:.2f}s per row)")
        
        # Save to output directory
        output_path = self._get_output_path(csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\nResults saved to: {output_path}")
        
        return df
    
    def _evaluate_reasoning_support(
        self, 
        reasoning: str, 
        predicted_label: str,
        article_title: str,
        article_content: str
    ) -> float:
        """Evaluate how well reasoning is supported by the article content.
        
        Args:
            reasoning: The classification reasoning to evaluate
            predicted_label: The predicted classification (fake/reliable)
            article_title: Title of the article
            article_content: Content of the article
            
        Returns:
            Support score between 0.0 and 1.0
        """
        # Construct prompt for LLM
        prompt = self._build_evaluation_prompt(reasoning, predicted_label, article_title, article_content)
        
        # Get LLM response
        try:
            response = self.llm.simple(
                prompt=prompt,
                system="You are an expert evaluator of logical reasoning and evidential support. You provide precise numerical scores.",
                temperature=0.0,
                max_tokens=256
            )
            
            # Parse score from response
            score = self._parse_score(response)
            
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            print(f"  Error evaluating reasoning: {e}")
            return 0.5  # Default to middle score on error
    
    def _build_evaluation_prompt(
        self, 
        reasoning: str, 
        predicted_label: str,
        article_title: str,
        article_content: str
    ) -> str:
        """Build prompt for LLM to evaluate reasoning support.
        
        Args:
            reasoning: The classification reasoning
            predicted_label: The predicted classification
            article_title: Title of the article
            article_content: Content of the article
            
        Returns:
            Prompt string for LLM
        """
        prompt = f"""Evaluate how well the REASONING is supported by the actual ARTICLE CONTENT below.

ARTICLE TITLE:
{article_title}

ARTICLE CONTENT:
{article_content}

PREDICTED CLASSIFICATION: {predicted_label}

REASONING:
{reasoning}

Rate the support on a scale from 0.0 to 1.0:
- 1.0 = Perfect support: reasoning is entirely based on and accurately reflects information from the article
- 0.7-0.9 = Good support: reasoning is mostly supported by the article with minor external inferences
- 0.4-0.6 = Moderate support: reasoning is partially supported by the article but includes significant external information or interpretation
- 0.1-0.3 = Poor support: reasoning is barely supported by the article content
- 0.0 = No support: reasoning is completely unrelated to or contradicts the article content

Provide ONLY a numerical score between 0.0 and 1.0 as your response.
Score:"""
        
        return prompt
    
    def _parse_score(self, response: str) -> float:
        """Parse numerical score from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed score as float
        """
        # Try to extract a number between 0 and 1
        # Look for patterns like "0.85", "0.7", "1.0", etc.
        match = re.search(r'\b([0-1]\.?\d*)\b', response.strip())
        
        if match:
            try:
                score = float(match.group(1))
                return score
            except ValueError:
                pass
        
        # If no valid number found, try to extract just digits
        match = re.search(r'(\d+)', response.strip())
        if match:
            try:
                num = int(match.group(1))
                # If it's a whole number like 85, assume it's out of 100
                if num > 1:
                    return num / 100.0
                return float(num)
            except ValueError:
                pass
        
        # Default to 0.5 if parsing fails
        print(f"  Warning: Could not parse score from response: {response[:100]}")
        return 0.5
    
    def _get_output_path(self, input_path: Path) -> Path:
        """Generate output path based on input path.
        
        Args:
            input_path: Input CSV file path
            
        Returns:
            Output path with format <original_dir>_reason_support/<original_filename>
        """
        # Get parent directory (e.g., "experiments")
        parent_dir = input_path.parent
        
        # If parent is relative, use it as base
        if not parent_dir.is_absolute():
            parent_name = parent_dir.name if parent_dir.name else "data"
            # Create new directory name (e.g., "experiments_reason_support")
            new_dir_name = f"{parent_name}_reason_support"
            new_dir = parent_dir.parent / new_dir_name
        else:
            # For absolute paths, create output dir at same level as input dir
            # e.g., STORAGE_DIR/experiments/file.csv -> STORAGE_DIR/experiments_reason_support/file.csv
            parent_name = parent_dir.name if parent_dir.name else "data"
            new_dir_name = f"{parent_name}_reason_support"
            new_dir = parent_dir.parent / new_dir_name
        
        # Keep the same filename
        output_path = new_dir / input_path.name
        
        return output_path


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning support from article content using LLM"
    )
    parser.add_argument("csv_path", help="Path to CSV file with classification results")
    default_articles_csv = str(Path(STORAGE_DIR) / "preprocessed" / "news_balanced.csv")
    parser.add_argument("--articles-csv", default=default_articles_csv,
                       help="Path to CSV file with article content")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator(articles_csv=args.articles_csv)
    
    # Evaluate reasoning
    df = evaluator.evaluate_csv(args.csv_path)
    
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()

