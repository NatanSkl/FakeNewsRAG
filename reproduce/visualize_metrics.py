#!/usr/bin/env python3
"""
Visualize metrics from JSON files in grouped histograms.

This script reads JSON files containing experiment metrics and creates
grouped bar charts for comparison across different experiments.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv('params.env')
STORAGE_DIR = os.getenv('STORAGE_DIR', '/StudentData/reproduce')


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(json_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant metrics from JSON data."""
    metrics = json_data.get('metrics', {})
    return {
        'f1': metrics.get('f1_score', 0.0),
        'accuracy': metrics.get('accuracy', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'mean_correct': metrics.get('reasoning_alignment_mean_correct', 0.0),
        'mean_incorrect': metrics.get('reasoning_alignment_mean_incorrect', 0.0)
    }


def get_experiment_name(filepath: str) -> str:
    """Extract a readable experiment name from the file path."""
    filename = os.path.basename(filepath)
    # Remove the _metrics.json suffix
    name = filename.replace('_metrics.json', '')
    return name


def extract_label_info(filename: str, group_name: str) -> str:
    """Extract relevant information from filename based on group."""
    # Remove the _metrics.json suffix
    name = filename.replace('_metrics.json', '')
    
    if 'Group1' in group_name:
        # Extract ce and diversity info
        # Parse the string to get key=value pairs
        parts = {}
        tokens = name.split('_')
        i = 0
        while i < len(tokens):
            if '=' in tokens[i]:
                key, value = tokens[i].split('=', 1)
                # Check if value continues in next tokens (until we hit another =)
                full_value = value
                i += 1
                while i < len(tokens) and '=' not in tokens[i]:
                    full_value += '_' + tokens[i]
                    i += 1
                parts[key] = full_value
            else:
                i += 1
        
        ce = parts.get('ce', 'None')
        diversity = parts.get('diversity', 'None')
        return f"ce={ce}, diversity={diversity}"
    
    elif 'Group2' in group_name:
        # Extract prompt and naming info
        parts = {}
        tokens = name.split('_')
        i = 0
        while i < len(tokens):
            if '=' in tokens[i]:
                key, value = tokens[i].split('=', 1)
                # Check if value continues in next tokens (until we hit another =)
                full_value = value
                i += 1
                while i < len(tokens) and '=' not in tokens[i]:
                    full_value += '_' + tokens[i]
                    i += 1
                parts[key] = full_value
            else:
                i += 1
        
        prompt = parts.get('prompt', '?')
        naming = parts.get('naming', '?')
        return f"prompt={prompt}, naming={naming}"
    
    elif 'Group3' in group_name:
        # For test experiments, show if it's baseline or RAG
        if 'llm_baseline' in name:
            return "LLM Baseline"
        else:
            # Extract prompt and naming for RAG
            parts = {}
            tokens = name.split('_')
            i = 0
            while i < len(tokens):
                if '=' in tokens[i]:
                    key, value = tokens[i].split('=', 1)
                    full_value = value
                    i += 1
                    while i < len(tokens) and '=' not in tokens[i]:
                        full_value += '_' + tokens[i]
                        i += 1
                    parts[key] = full_value
                else:
                    i += 1
            
            prompt = parts.get('prompt', '?')
            naming = parts.get('naming', '?')
            return f"RAG: prompt={prompt}, naming={naming}"
    
    else:
        # Default: return full name
        return name


def get_group_title(group_name: str) -> str:
    """Get the appropriate title for each group."""
    if 'Group1' in group_name:
        return "Metrics Comparison - Cross Encoder & MMR (Sampled Validation)"
    elif 'Group2' in group_name:
        return "Metrics Comparison - Prompt Types and Namings (Sampled Validation)"
    elif 'Group3' in group_name:
        return "Metrics Comparison - Baseline vs Best Config (Sampled Test)"
    else:
        return f"Metrics Comparison - {group_name}"


def find_matching_file(prefix: str, metrics_dir: str) -> str:
    """
    Find a metrics file that starts with the given prefix.
    
    Args:
        prefix: The prefix to match (without _metrics.json suffix)
        metrics_dir: Directory to search in
        
    Returns:
        Full filename of matching file, or None if not found
    """
    if not os.path.exists(metrics_dir):
        return None
    
    # Get all metrics files
    all_files = [f for f in os.listdir(metrics_dir) if f.endswith('_metrics.json')]
    
    # Find files that start with the prefix
    matches = [f for f in all_files if f.startswith(prefix)]
    
    if not matches:
        return None
    
    # If multiple matches, prefer the one without 'limit=' (or return first)
    # Sort to put files without 'limit=' first
    matches.sort(key=lambda x: ('limit=' in x, x))
    
    return matches[0]


def plot_grouped_metrics(groups: Dict[str, List[str]], output_dir: str, 
                         metrics_dir: str = None):
    """
    Create grouped bar charts for metrics comparison.
    
    Args:
        groups: Dictionary mapping group names to lists of metric file prefixes (without _metrics.json)
        output_dir: Directory to save the output plots
        metrics_dir: Directory containing the metrics JSON files (default: STORAGE_DIR/metrics_reason_support)
    """
    if metrics_dir is None:
        metrics_dir = str(Path(STORAGE_DIR) / "metrics_reason_support")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metric_names = ['f1', 'accuracy', 'precision', 'recall', 'mean_correct', 'mean_incorrect']
    
    for group_name, file_list in groups.items():
        # Load data for all experiments in this group
        experiments_data = []
        experiment_names = []
        filenames = []
        
        for file_prefix in file_list:
            # Remove _metrics.json if present (to get the prefix)
            prefix = file_prefix.replace('_metrics.json', '')
            
            # Find matching file
            filename = find_matching_file(prefix, metrics_dir)
            
            if filename is None:
                print(f"Warning: No file found starting with: {prefix}")
                continue
            
            # Show which file was matched
            if filename != f"{prefix}_metrics.json":
                print(f"  Matched '{prefix}' -> '{filename}'")
            
            filepath = os.path.join(metrics_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
            
            data = load_json_file(filepath)
            metrics = extract_metrics(data)
            experiments_data.append(metrics)
            experiment_names.append(extract_label_info(filename, group_name))
            filenames.append(filename)
        
        # Sort Group 2 by prompt type
        if 'Group2' in group_name:
            # Extract prompt numbers for sorting
            sort_data = []
            for i, filename in enumerate(filenames):
                # Extract prompt number
                prompt_num = -1
                for part in filename.split('_'):
                    if part.startswith('prompt='):
                        try:
                            prompt_num = int(part.split('=')[1])
                        except:
                            pass
                        break
                sort_data.append((prompt_num, experiments_data[i], experiment_names[i], filename))
            
            # Sort by prompt number
            sort_data.sort(key=lambda x: x[0])
            
            # Unpack sorted data
            experiments_data = [x[1] for x in sort_data]
            experiment_names = [x[2] for x in sort_data]
            filenames = [x[3] for x in sort_data]
        
        if not experiments_data:
            print(f"Warning: No data found for group {group_name}")
            continue
        
        # Create the grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(experiments_data)  # Width of bars
        
        # Plot bars for each experiment
        for i, (exp_data, exp_name) in enumerate(zip(experiments_data, experiment_names)):
            offset = (i - len(experiments_data) / 2) * width + width / 2
            values = [exp_data[metric] for metric in metric_names]
            ax.bar(x + offset, values, width, label=exp_name)
        
        # Customize the chart
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title(get_group_title(group_name), fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{group_name}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()
        
        # Also create a table of values
        create_metrics_table(experiments_data, experiment_names, metric_names, 
                           group_name, output_dir)


def create_metrics_table(experiments_data: List[Dict[str, float]], 
                        experiment_names: List[str],
                        metric_names: List[str],
                        group_name: str,
                        output_dir: str):
    """Create a text table of metrics for easy reference."""
    output_path = os.path.join(output_dir, f'{group_name}_metrics_table.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"{get_group_title(group_name)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Experiment':<50} ")
        for metric in metric_names:
            f.write(f"{metric:<15} ")
        f.write("\n")
        f.write("-" * 150 + "\n")
        
        # Data rows
        for exp_name, exp_data in zip(experiment_names, experiments_data):
            # Truncate long names
            display_name = exp_name[:47] + "..." if len(exp_name) > 50 else exp_name
            f.write(f"{display_name:<50} ")
            for metric in metric_names:
                f.write(f"{exp_data[metric]:<15.4f} ")
            f.write("\n")
    
    print(f"Saved metrics table to: {output_path}")


def get_all_files_except(directory: str, exclude_pattern: str) -> List[str]:
    """Get all JSON files in a directory except those matching the exclude pattern."""
    all_files = [f for f in os.listdir(directory) if f.endswith('_metrics.json')]
    return [f for f in all_files if exclude_pattern not in f]


def main():
    """Main function to define groups and create visualizations."""
    metrics_dir = str(Path(STORAGE_DIR) / "metrics_reason_support")
    output_dir = str(Path(STORAGE_DIR) / "visualized")
    
    # Define groups of experiments
    groups = {
        'Group1_Baseline_and_MMR': [
            'val_sampled_rag_ce=None_diversity=None_prompt=0_naming=fake_reliable_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=0_naming=fake_reliable_metrics.json'
        ],
        'Group2_Prompt_and_Naming': [
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=0_naming=fake_reliable_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=0_naming=type1_type2_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=1_naming=fake_reliable_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=1_naming=type1_type2_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=2_naming=fake_reliable_metrics.json',
            'val_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=2_naming=type1_type2_metrics.json'
        ],
        'Group3_Test_Results': [
            'test_sampled_llm_baseline_metrics.json',
            'test_sampled_rag_ce=ms-marco-MiniLM-L-6-v2_diversity=mmr_prompt=2_naming=fake_reliable_metrics.json'
        ]
    }
    
    print("Starting visualization...")
    print(f"Metrics directory: {metrics_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Show what prefixes are in each group
    for group_name, file_list in groups.items():
        print(f"\n{group_name}:")
        for f in file_list:
            prefix = f.replace('_metrics.json', '')
            print(f"  - Looking for files starting with: {prefix}")
    
    print("\n" + "="*80 + "\n")
    
    # Create visualizations
    plot_grouped_metrics(groups, output_dir, metrics_dir)
    
    print("\n" + "="*80)
    print("Visualization complete!")


if __name__ == '__main__':
    main()

