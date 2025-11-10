#!/usr/bin/env python3
"""
Run reasoning support evaluations on all CSV files in experiments directory.
Evaluates how well reasoning is supported by the actual article content.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import from evaluate
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate.reasoning_evaluator import ReasoningEvaluator

def main():
    # Load STORAGE_DIR from params.env
    load_dotenv('params.env')
    STORAGE_DIR = os.getenv('STORAGE_DIR', '/StudentData/reproduce')
    
    # Directory containing files to evaluate
    eval_dir = Path(STORAGE_DIR) / "experiments"
    
    # Create evaluator
    print("Initializing evaluator...")
    evaluator = ReasoningEvaluator()
    
    # Find all CSV files
    csv_files = sorted(eval_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files to evaluate")
    print("="*80)
    
    # Evaluate each file
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Evaluating: {csv_path.name}")
        print("="*80)
        
        try:
            results = evaluator.evaluate_csv(str(csv_path))
            print(f"✓ Successfully evaluated {csv_path.name}")
        except Exception as e:
            print(f"✗ Error evaluating {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETE")
    print("="*80)
    output_dir = Path(STORAGE_DIR) / "experiments_reason_support"
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()

