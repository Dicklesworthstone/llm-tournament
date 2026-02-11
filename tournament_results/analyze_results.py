#!/usr/bin/env python3
"""
LLM Tournament Results Analyzer

This script analyzes and visualizes the results of the LLM tournament.
It creates plots and tables comparing the performance of different models
across rounds.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Optional import for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_metrics(metrics_dir: str) -> Dict[str, Any]:
    """Load metrics from JSON files"""
    metrics_dir = Path(metrics_dir)
    
    # Load tournament metrics
    tournament_metrics_path = metrics_dir / "tournament_metrics.json"
    tournament_metrics = {}
    
    if tournament_metrics_path.exists():
        with open(tournament_metrics_path, "r", encoding="utf-8") as f:
            tournament_metrics = json.load(f)
    
    # Load test metrics
    test_metrics_path = metrics_dir / "test_metrics.json"
    test_metrics = []
    
    if test_metrics_path.exists():
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
    
    return {
        "tournament": tournament_metrics,
        "test": test_metrics
    }
    
def generate_markdown_report(metrics: Dict[str, Any], output_file: str) -> None:
    """Generate a markdown report from the metrics"""
    tournament_metrics = metrics.get("tournament", {})
    test_metrics = metrics.get("test", [])
    
    # Start building the report
    report = "# LLM Tournament Results\n\n"
    report += "## Overview\n\n"
    report += "This report summarizes the results of the LLM tournament.\n\n"
    
    # Add test metrics table if available
    if test_metrics:
        report += "## Test Results\n\n"
        report += "| Model | Round | Execution Time (s) | Output Lines | Output Size (KB) |\n"
        report += "|-------|-------|-------------------|--------------|------------------|\n"
        
        # Sort by round, then model
        sorted_metrics = sorted(test_metrics, key=lambda x: (int(x.get("round", 0)), x.get("model", "")))
        
        for metric in sorted_metrics:
            model = metric.get("model", "Unknown")
            round_num = metric.get("round", "Unknown")
            time = metric.get("execution_time", "N/A")
            lines = metric.get("output_lines", "N/A")
            size = metric.get("output_size_kb", "N/A")
            
            report += f"| {model} | {round_num} | {time} | {lines} | {size} |\n"
            
    # Add round metrics if available
    rounds_data = tournament_metrics.get("rounds", {})
    if rounds_data:
        report += "\n## Code Metrics by Round\n\n"
        report += "| Round | Model | Code Size (KB) | Code Lines |\n"
        report += "|-------|-------|---------------|------------|\n"
        
        for round_num, round_data in sorted(rounds_data.items(), key=lambda x: int(x[0])):
            models_data = round_data.get("models", {})
            
            for model, model_data in sorted(models_data.items()):
                code_size = model_data.get("code_size_kb", "N/A")
                code_lines = model_data.get("code_lines", "N/A")
                
                report += f"| {round_num} | {model} | {code_size} | {code_lines} |\n"
    
    # Save the report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")

def create_visualizations(metrics: Dict[str, Any], output_dir: str) -> None:
    """Create visualizations of the metrics"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualizations.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    tournament_metrics = metrics.get("tournament", {})
    test_metrics = metrics.get("test", [])
    
    # Extract data for plotting
    rounds_data = tournament_metrics.get("rounds", {})
    if not rounds_data:
        return
    
    # Create a plot of code size by round and model
    plt.figure(figsize=(10, 6))
    
    # Extract data
    models = set()
    round_nums = []
    data_by_model = {}
    
    for round_num, round_data in sorted(rounds_data.items(), key=lambda x: int(x[0])):
        round_nums.append(int(round_num))
        models_data = round_data.get("models", {})
        
        for model, model_data in models_data.items():
            models.add(model)
            if model not in data_by_model:
                data_by_model[model] = []
            
            code_size = model_data.get("code_size_kb", 0)
            data_by_model[model].append(code_size)
    
    # Plot code size by round for each model
    for model, sizes in data_by_model.items():
        # Ensure all models have data for all rounds
        while len(sizes) < len(round_nums):
            sizes.append(None)
        
        plt.plot(round_nums, sizes, marker='o', label=model)
    
    plt.xlabel('Round')
    plt.ylabel('Code Size (KB)')
    plt.title('Code Size by Round and Model')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(output_dir / "code_size_by_round.png")
    
    # Create a plot of execution time by round and model
    if test_metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        data_by_model = {}
        
        for metric in test_metrics:
            model = metric.get("model", "Unknown")
            round_num = int(metric.get("round", 0))
            time = metric.get("execution_time", 0)
            
            if model not in data_by_model:
                data_by_model[model] = []
            
            # Ensure the list is long enough
            while len(data_by_model[model]) <= round_num:
                data_by_model[model].append(None)
            
            data_by_model[model][round_num] = time
        
        # Plot execution time by round for each model
        for model, times in data_by_model.items():
            rounds = list(range(len(times)))
            plt.plot(rounds, times, marker='o', label=model)
        
        plt.xlabel('Round')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time by Round and Model')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(output_dir / "execution_time_by_round.png")
    
    print("Visualizations saved to: {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze LLM tournament results")
    parser.add_argument("--metrics-dir", type=str, default="metrics",
                     help="Directory containing metrics files")
    parser.add_argument("--output-dir", type=str, default="analysis",
                     help="Directory for output files")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load metrics
    metrics = load_metrics(args.metrics_dir)
    
    # Generate markdown report
    report_path = output_dir / "tournament_results_report.md"
    generate_markdown_report(metrics, report_path)
    
    # Create visualizations
    create_visualizations(metrics, output_dir)

if __name__ == "__main__":
    main()
