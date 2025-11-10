#!/usr/bin/env python3
"""
Run Evaluations on All CSV Files

Runs the evaluator on every CSV file in experiments_reason_support
and outputs metrics to metrics_reason_support.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path to import from evaluate
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.evaluator import Evaluator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run evaluator on all CSV files in experiments_reason_support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluations.py
  python run_evaluations.py --experiments-dir /path/to/experiments --metrics-dir /path/to/metrics
        """
    )
    
    # Load environment variables
    load_dotenv('params.env')
    STORAGE_DIR = os.getenv('STORAGE_DIR', '/StudentData/reproduce')
    
    parser.add_argument(
        "--experiments-dir",
        default=os.path.join(STORAGE_DIR, "experiments_reason_support"),
        help="Directory containing CSV files to evaluate (default: STORAGE_DIR/experiments_reason_support)"
    )
    parser.add_argument(
        "--metrics-dir",
        default=os.path.join(STORAGE_DIR, "metrics_reason_support"),
        help="Directory to save metrics (default: STORAGE_DIR/metrics_reason_support)"
    )
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    metrics_dir = Path(args.metrics_dir)
    
    print("="*80)
    print("BATCH EVALUATION RUNNER")
    print("="*80)
    print(f"Experiments directory: {experiments_dir}")
    print(f"Metrics directory: {metrics_dir}")
    print()
    
    # Check if experiments directory exists
    if not experiments_dir.exists():
        print(f"❌ Error: Experiments directory does not exist: {experiments_dir}")
        return 1
    
    # Create metrics directory
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(experiments_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  Warning: No CSV files found in {experiments_dir}")
        return 0
    
    print(f"Found {len(csv_files)} CSV file(s) to evaluate:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    print()
    
    # Initialize evaluator
    evaluator = Evaluator(metrics_dir=str(metrics_dir))
    
    # Evaluate each CSV file
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*80}")
        print(f"Evaluating file {i}/{len(csv_files)}: {csv_file.name}")
        print(f"{'='*80}")
        
        try:
            results, metrics = evaluator.evaluate_from_csv(str(csv_file))
            print(f"✅ Successfully evaluated: {csv_file.name}")
            successful += 1
        except Exception as e:
            print(f"❌ Error evaluating {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(csv_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Metrics saved to: {metrics_dir}")
    print()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

