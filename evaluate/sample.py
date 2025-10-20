#!/usr/bin/env python3
"""
CSV Sampling Script

Samples a random subset from a CSV file and saves it to a new file.
"""

import argparse
import pandas as pd
import random
from pathlib import Path


def sample_csv(input_path: str, output_path: str, n_samples: int, random_seed: int = 42):
    """
    Sample n random rows from a CSV file and save to a new CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        n_samples: Number of samples to take
        random_seed: Random seed for reproducibility (optional)
    """
    print(f"Loading CSV from: {input_path}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Check for label column
    label_col = None
    if 'label' in df.columns:
        label_col = 'label'
    elif 'type' in df.columns:
        label_col = 'type'
    
    if label_col is None:
        print("Warning: No label column found, using random sampling")
        # Fall back to random sampling
        if len(df) < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {len(df)} rows available")
            print(f"Using all {len(df)} rows")
            n_samples = len(df)
        
        # Set random seed
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
        
        # Sample random rows
        print(f"Sampling {n_samples} random rows...")
        sampled_df = df.sample(n=n_samples, random_state=random_seed)
    else:
        # Balanced sampling by label
        print(f"Performing balanced sampling: {n_samples//2} reliable + {n_samples//2} fake")
        
        # Set random seed
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
        
        # Get reliable and fake articles
        reliable_df = df[df[label_col].str.lower().str.contains('reliable|credible|real', na=False)]
        fake_df = df[df[label_col].str.lower().str.contains('fake', na=False)]
        
        print(f"Found {len(reliable_df)} reliable articles, {len(fake_df)} fake articles")
        
        # Calculate samples per category
        n_reliable = n_samples // 2
        n_fake = n_samples - n_reliable  # Handle odd numbers
        
        # Check if we have enough samples
        if len(reliable_df) < n_reliable:
            print(f"Warning: Only {len(reliable_df)} reliable articles available, using all")
            n_reliable = len(reliable_df)
        
        if len(fake_df) < n_fake:
            print(f"Warning: Only {len(fake_df)} fake articles available, using all")
            n_fake = len(fake_df)
        
        # Sample from each category
        print(f"Sampling {n_reliable} reliable and {n_fake} fake articles...")
        sampled_reliable = reliable_df.sample(n=n_reliable, random_state=random_seed)
        sampled_fake = fake_df.sample(n=n_fake, random_state=random_seed)
        
        # Combine and shuffle
        sampled_df = pd.concat([sampled_reliable, sampled_fake]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save to output file
    print(f"Saving sampled data to: {output_path}")
    try:
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the sampled data
        sampled_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(sampled_df)} rows to {output_path}")
        
        # Show some basic statistics
        if 'type' in sampled_df.columns or 'label' in sampled_df.columns:
            label_col = 'type' if 'type' in sampled_df.columns else 'label'
            print(f"\nSample distribution:")
            print(sampled_df[label_col].value_counts())
        
        return True
        
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Sample random rows from a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sample.py /StudentData/preprocessed/val.csv 200 /StudentData/preprocessed/val_sampled.csv
  python sample.py data/train.csv 1000 output/sampled_train.csv --seed 42
        """
    )
    
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("n_samples", type=int, help="Number of samples to take")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_csv).exists():
        print(f"Error: Input file does not exist: {args.input_csv}")
        return 1
    
    # Validate sample size
    if args.n_samples <= 0:
        print(f"Error: Sample size must be positive, got: {args.n_samples}")
        return 1
    
    print("="*80)
    print("CSV SAMPLING SCRIPT")
    print("="*80)
    print(f"Input file: {args.input_csv}")
    print(f"Sample size: {args.n_samples}")
    print(f"Output file: {args.output_csv}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Run the sampling
    success = sample_csv(
        input_path=args.input_csv,
        output_path=args.output_csv,
        n_samples=args.n_samples,
        random_seed=args.seed
    )
    
    if success:
        print("\n" + "="*80)
        print("SAMPLING COMPLETED SUCCESSFULLY")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("SAMPLING FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
