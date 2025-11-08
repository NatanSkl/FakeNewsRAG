#!/usr/bin/env python3
"""
Run Both Configurations Script

Runs both RAG and LLM-only configurations sequentially on test_sampled.csv.
Useful for comparing RAG vs LLM-only baseline performance.

Example usage:
    /StudentData/rag2/bin/python evaluate/run_both_configs.py
    /StudentData/rag2/bin/python evaluate/run_both_configs.py --limit 10
    /StudentData/rag2/bin/python evaluate/run_both_configs.py --rag-preset best
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.rag_executor import RAGExecutor
from evaluate.only_llm_executor import OnlyLLMExecutor
from retrieval.retrieval_v3 import RetrievalConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run both RAG and LLM-only configurations sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both with default best config
  /StudentData/rag2/bin/python evaluate/run_both_configs.py
  
  # Test on 10 articles
  /StudentData/rag2/bin/python evaluate/run_both_configs.py --limit 10
  
  # Use baseline RAG config
  /StudentData/rag2/bin/python evaluate/run_both_configs.py --rag-preset baseline
        """
    )
    
    # Input/output paths
    parser.add_argument(
        "--input-csv",
        type=str,
        default="/StudentData/preprocessed/test_sampled.csv",
        help="Path to input CSV file (default: test_sampled.csv)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for results (default: experiments)"
    )
    
    # RAG configuration preset
    parser.add_argument(
        "--rag-preset",
        type=str,
        default="best",
        choices=["best", "baseline", "prompt0", "prompt1"],
        help="RAG configuration preset: 'best' (Prompt 2, CE+MMR), 'baseline' (no CE/MMR), 'prompt0', 'prompt1'"
    )
    
    # Server configuration
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://127.0.0.1:8010",
        help="LLM server URL (default: http://127.0.0.1:8010)"
    )
    
    parser.add_argument(
        "--store-path",
        type=str,
        default="/StudentData/index",
        help="Base path for vector stores (default: /StudentData/index)"
    )
    
    # Processing options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to process (for testing)"
    )
    
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG execution (only run LLM)"
    )
    
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM execution (only run RAG)"
    )
    
    return parser.parse_args()


def get_rag_config(preset: str):
    """Get RAG configuration based on preset name."""
    configs = {
        "best": {
            "ce_model": "ms-marco-MiniLM-L-6-v2",
            "diversity": "mmr",
            "prompt": 2,
            "naming": "fake_reliable",
            "description": "Best performing config (69% accuracy, Prompt 2, CE+MMR)"
        },
        "baseline": {
            "ce_model": None,
            "diversity": None,
            "prompt": 0,
            "naming": "fake_reliable",
            "description": "RAG baseline (no CE/MMR, Prompt 0)"
        },
        "prompt0": {
            "ce_model": "ms-marco-MiniLM-L-6-v2",
            "diversity": "mmr",
            "prompt": 0,
            "naming": "fake_reliable",
            "description": "Prompt 0 with CE+MMR"
        },
        "prompt1": {
            "ce_model": "ms-marco-MiniLM-L-6-v2",
            "diversity": "mmr",
            "prompt": 1,
            "naming": "fake_reliable",
            "description": "Prompt 1 with CE+MMR"
        }
    }
    return configs[preset]


def run_rag(args, config):
    """Run RAG pipeline."""
    print("\n" + "="*80)
    print("STEP 1: RUNNING RAG CONFIGURATION")
    print("="*80)
    print(f"Preset: {args.rag_preset}")
    print(f"Description: {config['description']}")
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Cross-Encoder: {config['ce_model'] or 'None'}")
    print(f"Diversity: {config['diversity'] or 'None'}")
    print(f"Prompt Type: {config['prompt']}")
    print(f"Naming: {config['naming']}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print("="*80)
    print()
    
    # Create retrieval config
    retrieval_config = RetrievalConfig(
        k=10,
        ce_model_name=config['ce_model'],
        diversity_type=config['diversity'],
        verbose=False
    )
    
    # Initialize executor
    executor = RAGExecutor(
        llm_url=args.llm_url,
        store_path=args.store_path,
        debug_mode=True
    )
    
    # Initialize (connect to LLM, load stores)
    print("Initializing RAG executor...")
    executor.initialize()
    
    # Run pipeline
    output_path = executor.run_rag_pipeline(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
        retrieval_config=retrieval_config,
        prompt_type=config['prompt'],
        naming_convention=config['naming'],
        limit=args.limit,
        fix_missing=False
    )
    
    print()
    print("="*80)
    print("RAG EXECUTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print()
    
    return output_path


def run_llm(args):
    """Run LLM-only baseline."""
    print("\n" + "="*80)
    print("STEP 2: RUNNING LLM-ONLY BASELINE")
    print("="*80)
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print("="*80)
    print()
    
    # Initialize executor
    executor = OnlyLLMExecutor(llm_url=args.llm_url)
    
    # Initialize (connect to LLM)
    print("Initializing LLM executor...")
    executor.initialize()
    
    # Run classification
    output_path = executor.run_llm_baseline(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    print()
    print("="*80)
    print("LLM EXECUTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print()
    
    return output_path


def main():
    args = parse_args()
    
    # Validate input file exists
    if not Path(args.input_csv).exists():
        print(f"ERROR: Input file does not exist: {args.input_csv}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get RAG configuration
    rag_config = get_rag_config(args.rag_preset)
    
    print("\n" + "="*80)
    print("RUNNING BOTH CONFIGURATIONS")
    print("="*80)
    print(f"Will execute: {'RAG' if not args.skip_rag else ''}{' and ' if not args.skip_rag and not args.skip_llm else ''}{'LLM-only' if not args.skip_llm else ''}")
    print(f"Input: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    results = {}
    
    try:
        # Run RAG
        if not args.skip_rag:
            rag_output = run_rag(args, rag_config)
            results['rag'] = rag_output
        else:
            print("\nSkipping RAG execution (--skip-rag)")
        
        # Run LLM-only
        if not args.skip_llm:
            llm_output = run_llm(args)
            results['llm'] = llm_output
        else:
            print("\nSkipping LLM execution (--skip-llm)")
        
        # Final summary
        print("\n" + "="*80)
        print("ALL EXECUTIONS COMPLETE")
        print("="*80)
        for executor_type, output_path in results.items():
            print(f"{executor_type.upper():10s}: {output_path}")
        print("="*80)
        print("\n✓ Both configurations executed successfully!")
        print("\nNext steps:")
        print("  1. Run evaluator on both outputs to calculate metrics")
        print("  2. Compare performance between RAG and LLM-only")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

