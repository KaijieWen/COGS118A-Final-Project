#!/usr/bin/env python3
"""
Check if experiments are complete and run analysis if so.
"""

import os
import pandas as pd
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(project_root, 'results', 'experiment_results.csv')

if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    expected_experiments = 81  # 3 datasets × 3 classifiers × 3 partitions × 3 trials
    
    if len(df) >= expected_experiments:
        print(f"✓ Experiments complete: {len(df)}/{expected_experiments} results found")
        print("\nRunning analysis...")
        sys.path.insert(0, os.path.join(project_root, 'src'))
        from analyze_results import main as analyze_main
        analyze_main()
    else:
        print(f"Experiments in progress: {len(df)}/{expected_experiments} completed")
        print("Please wait for experiments to complete...")
else:
    print("Experiments not started or results file not found.")
    print("Run: python src/run_experiments.py")


