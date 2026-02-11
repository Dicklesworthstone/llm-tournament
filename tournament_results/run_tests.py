#!/usr/bin/env python3
"""
LLM Tournament Test Runner

This script runs the test suite on the provided test file.
"""

import subprocess
import argparse
from pathlib import Path

def main():
    """Run the test suite"""
    parser = argparse.ArgumentParser(description="Run LLM tournament tests")
    parser.add_argument("--test-file", type=str, required=True, help="File to test on")
    args = parser.parse_args()
    
    # Path to the test script
    test_script = Path(__file__).parent / "test_all_solutions.py"
    
    # Run the test script
    cmd = [
        "python",
        str(test_script),
        "--input", args.test_file,
        "--output-dir", "output_results_for_each_round_and_model"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()
