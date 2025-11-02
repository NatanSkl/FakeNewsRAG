"""
Experiment Runner

Runs multiple RAG experiments with different configurations using cartesian product
of parameters. Handles errors gracefully and tracks experiment results.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import product

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.rag_executor import RAGExecutor
from retrieval.retrieval_v3 import RetrievalConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    retrieval_config: RetrievalConfig
    prompt_type: int
    naming_convention: str
    config_name: str  # Human-readable name for this config
    
    def __str__(self):
        return self.config_name


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""
    config: ExperimentConfig
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    duration: float = 0.0


class ExperimentRunner:
    """
    Runs multiple RAG experiments with different configurations.
    
    Generates cartesian product of all parameter combinations and runs
    each experiment using RAGExecutor.
    """
    
    def __init__(
        self,
        llm_url: str = "http://127.0.0.1:8010",
        store_path: str = "/StudentData/index",
        debug_mode: bool = True
    ):
        """
        Initialize the ExperimentRunner.
        
        Args:
            llm_url: URL of the LLM server
            store_path: Base path to vector stores
            debug_mode: Whether to save debug information (prompts)
        """
        self.llm_url = llm_url
        self.store_path = store_path
        self.debug_mode = debug_mode
        self.executor = None
        
        print("="*80)
        print("EXPERIMENT RUNNER")
        print("="*80)
        print("Runs multiple RAG experiments with different configurations")
        print(f"LLM URL: {llm_url}")
        print(f"Store path: {store_path}")
        print(f"Debug mode: {debug_mode}")
        print()
    
    def initialize(self):
        """Initialize the RAG executor."""
        print("INITIALIZING RAG EXECUTOR")
        print("-"*80)
        
        self.executor = RAGExecutor(
            llm_url=self.llm_url,
            store_path=self.store_path,
            debug_mode=self.debug_mode
        )
        self.executor.initialize()
        
        print("Initialization complete!")
        print()
    
    def generate_experiments(
        self,
        retrieval_configs: List[str],
        prompt_types: List[int],
        naming_conventions: List[str]
    ) -> List[ExperimentConfig]:
        """
        Generate all experiment configurations as cartesian product of parameters.
        
        Args:
            retrieval_configs: List of retrieval config types: "default", "optimized", or both
            prompt_types: List of prompt types (e.g., [0, 1, 2])
            naming_conventions: List of naming conventions (e.g., ["fake_reliable", "type1_type2"])
        
        Returns:
            List of ExperimentConfig objects
        """
        print("GENERATING EXPERIMENT CONFIGURATIONS")
        print("-"*80)
        
        # Generate retrieval configurations
        retrieval_config_objects = []
        for config_type in retrieval_configs:
            if config_type == "default":
                config = RetrievalConfig(
                    k=8,
                    ce_model_name=None,
                    diversity_type=None,
                    verbose=False
                )
                retrieval_config_objects.append((config_type, config))
            elif config_type == "optimized":
                config = RetrievalConfig(
                    k=8,
                    ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    diversity_type="mmr",
                    verbose=False
                )
                retrieval_config_objects.append((config_type, config))
            else:
                print(f"WARNING: Unknown retrieval config type: {config_type}")
        
        # Generate cartesian product of all parameters
        experiments = []
        for (ret_name, ret_config), prompt_type, naming_conv in product(
            retrieval_config_objects,
            prompt_types,
            naming_conventions
        ):
            # Create human-readable config name
            config_name = f"retrieval={ret_name}_prompt={prompt_type}_naming={naming_conv}"
            
            experiment = ExperimentConfig(
                retrieval_config=ret_config,
                prompt_type=prompt_type,
                naming_convention=naming_conv,
                config_name=config_name
            )
            experiments.append(experiment)
        
        print(f"Generated {len(experiments)} experiment configurations:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp.config_name}")
        print()
        
        return experiments
    
    def run_experiments(
        self,
        csv_path: str,
        output_dir: str,
        retrieval_configs: List[str] = ["default"],
        prompt_types: List[int] = [0],
        naming_conventions: List[str] = ["fake_reliable"],
        limit: Optional[int] = None,
        fix_missing: bool = False
    ) -> List[ExperimentResult]:
        """
        Run all experiments defined by the cartesian product of parameters.
        
        Args:
            csv_path: Path to input CSV file
            output_dir: Directory to save results
            retrieval_configs: List of retrieval config types ("default", "optimized")
            prompt_types: List of prompt types to test
            naming_conventions: List of naming conventions to test
            limit: Maximum number of articles to process per experiment
            fix_missing: Whether to only process missing results
        
        Returns:
            List of ExperimentResult objects
        """
        # Ensure executor is initialized
        if self.executor is None:
            self.initialize()
        
        # Generate all experiment configurations
        experiments = self.generate_experiments(
            retrieval_configs=retrieval_configs,
            prompt_types=prompt_types,
            naming_conventions=naming_conventions
        )
        
        # Run each experiment
        results = []
        
        print("="*80)
        print(f"RUNNING {len(experiments)} EXPERIMENTS")
        print("="*80)
        print()
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(experiments)}: {experiment.config_name}")
            print(f"{'='*80}")
            
            start_time = time.time()
            
            try:
                # Run the experiment
                output_path = self.executor.run_rag_pipeline(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    retrieval_config=experiment.retrieval_config,
                    prompt_type=experiment.prompt_type,
                    naming_convention=experiment.naming_convention,
                    limit=limit,
                    fix_missing=fix_missing
                )
                
                duration = time.time() - start_time
                
                result = ExperimentResult(
                    config=experiment,
                    success=True,
                    output_path=output_path,
                    duration=duration
                )
                
                print(f"\n✓ SUCCESS: Completed in {duration:.1f}s")
                print(f"  Output: {output_path}")
                
            except Exception as e:
                duration = time.time() - start_time
                
                result = ExperimentResult(
                    config=experiment,
                    success=False,
                    error_message=str(e),
                    duration=duration
                )
                
                print(f"\n✗ FAILED: {e}")
                print(f"  Duration before failure: {duration:.1f}s")
                
                # Print traceback for debugging
                import traceback
                print("\nError traceback:")
                traceback.print_exc()
            
            results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[ExperimentResult]):
        """Print summary of all experiment results."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\nTotal experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n✓ SUCCESSFUL EXPERIMENTS:")
            for result in successful:
                print(f"  - {result.config.config_name}")
                print(f"    Duration: {result.duration:.1f}s")
                print(f"    Output: {result.output_path}")
        
        if failed:
            print("\n✗ FAILED EXPERIMENTS:")
            for result in failed:
                print(f"  - {result.config.config_name}")
                print(f"    Error: {result.error_message}")
        
        total_time = sum(r.duration for r in results)
        avg_time = total_time / len(results) if results else 0
        
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"Average time per experiment: {avg_time:.1f}s")
        
        print("="*80)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run multiple RAG experiments with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List experiments without running them
  python experiment_runner.py data.csv --retrieval-configs default optimized --prompt-types 0 1 2 --list-experiments
  
  # Run with default retrieval and prompt types 0,1,2
  python experiment_runner.py data.csv --retrieval-configs default --prompt-types 0 1 2
  
  # Run with both retrieval configs and all naming conventions
  python experiment_runner.py data.csv --retrieval-configs default optimized --naming-conventions fake_reliable type1_type2
  
  # Run complete experiment suite with limit for testing
  python experiment_runner.py data.csv --retrieval-configs default optimized --prompt-types 0 1 2 --naming-conventions fake_reliable type1_type2 --limit 10
        """
    )
    
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="experiments", help="Output directory")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8010", help="LLM server URL")
    parser.add_argument("--store-path", default="/StudentData/index", help="Path to store directory")
    
    # Experiment parameters
    parser.add_argument(
        "--retrieval-configs",
        nargs="+",
        choices=["default", "optimized"],
        default=["default"],
        help="Retrieval configurations to test"
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        type=int,
        default=[0],
        help="Prompt types to test (e.g., 0 1 2)"
    )
    parser.add_argument(
        "--naming-conventions",
        nargs="+",
        choices=["fake_reliable", "type1_type2"],
        default=["fake_reliable"],
        help="Naming conventions to test"
    )
    
    # Execution parameters
    parser.add_argument("--limit", type=int, help="Limit number of articles per experiment (for testing)")
    parser.add_argument("--fix-missing", action="store_true", help="Only process rows with missing results")
    parser.add_argument("--debug-mode", action="store_true", default=True, help="Enable debug mode")
    parser.add_argument("--list-experiments", action="store_true", help="List experiments without running them")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(
        llm_url=args.llm_url,
        store_path=args.store_path,
        debug_mode=args.debug_mode
    )
    
    # If --list-experiments flag is set, just show what will run
    if args.list_experiments:
        print("="*80)
        print("LISTING EXPERIMENTS (not running)")
        print("="*80)
        print()
        
        experiments = runner.generate_experiments(
            retrieval_configs=args.retrieval_configs,
            prompt_types=args.prompt_types,
            naming_conventions=args.naming_conventions
        )
        
        print(f"\nTotal experiments to run: {len(experiments)}\n")
        
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp.config_name}")
            print(f"   - Retrieval: k={exp.retrieval_config.k}, "
                  f"CE={exp.retrieval_config.ce_model_name or 'None'}, "
                  f"Diversity={exp.retrieval_config.diversity_type or 'None'}")
            print(f"   - Prompt type: {exp.prompt_type}")
            print(f"   - Naming convention: {exp.naming_convention}")
            print()
        
        print("="*80)
        print(f"To run these {len(experiments)} experiments, remove the --list-experiments flag")
        print("="*80)
        
        sys.exit(0)
    
    # Run experiments
    results = runner.run_experiments(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        retrieval_configs=args.retrieval_configs,
        prompt_types=args.prompt_types,
        naming_conventions=args.naming_conventions,
        limit=args.limit,
        fix_missing=args.fix_missing
    )
    
    # Exit with error code if any experiments failed
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        print(f"\nWarning: {failed_count} experiment(s) failed")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

