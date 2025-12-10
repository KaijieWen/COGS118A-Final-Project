"""
Master experiment loop: 3 datasets × 3 partitions × 3 trials × 3 classifiers = 81 runs.
Performs hyperparameter tuning and evaluation for each combination.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import json
import signal
import time

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from load_adult import load_adult_dataset
from load_mushroom import load_mushroom_dataset
from load_spambase import load_spambase_dataset
from data_splits import get_train_test_split, get_partition_ratios, get_trial_seeds
from hyperparameter_tuning import tune_hyperparameters

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

# Global timeout handler
timeout_occurred = False

def timeout_handler(signum, frame):
    global timeout_occurred
    timeout_occurred = True
    raise TimeoutError("Experiment timed out")

def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: 'adult', 'spambase', or 'mushroom'
        
    Returns:
        X, y: Feature matrix and target vector
    """
    loaders = {
        'adult': load_adult_dataset,
        'spambase': load_spambase_dataset,
        'mushroom': load_mushroom_dataset
    }
    
    if dataset_name == 'spambase':
        X, y, _ = loaders[dataset_name]()
        return X, y
    else:
        X, y = loaders[dataset_name]()
        return X, y

def run_single_experiment(dataset_name, classifier_name, partition_name, train_ratio, trial_seed, timeout=600):
    """
    Run a single experiment: load data, split, tune, train, evaluate.
    
    Args:
        dataset_name: 'adult', 'spambase', or 'mushroom'
        classifier_name: 'random_forest', 'svm', or 'knn'
        partition_name: '20_80', '50_50', or '80_20'
        train_ratio: Training ratio (0.2, 0.5, or 0.8)
        trial_seed: Random seed for this trial
        timeout: Maximum time in seconds (default: 600 = 10 minutes)
        
    Returns:
        Dictionary with results
    """
    global timeout_occurred
    timeout_occurred = False
    
    start_time = time.time()
    
    try:
        # Load dataset
        X, y = load_dataset(dataset_name)
        
        # Split data
        X_train, X_test, y_train, y_test = get_train_test_split(
            X, y, train_ratio, random_state=trial_seed
        )
        
        # For SVM on large datasets, always use LinearSVC with sampling for speed
        # This is acceptable for empirical comparison purposes
        use_linear_svm = False
        if classifier_name == 'svm' and X_train.shape[0] > 5000:
            from sklearn.model_selection import train_test_split
            # Always use LinearSVC for datasets > 5000 samples (much faster)
            use_linear_svm = True
            # Sample 3,000 training samples for LinearSVC (faster)
            sample_size = min(3000, X_train.shape[0])
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=sample_size, random_state=trial_seed, stratify=y_train
            )
            print(f"  Using LinearSVC with sample of {X_train_sample.shape[0]} (from {X_train.shape[0]})")
            X_train_tune = X_train_sample
            y_train_tune = y_train_sample
        else:
            X_train_tune = X_train
            y_train_tune = y_train
        
        # Tune hyperparameters (5-fold CV on training set)
        best_model, best_params, val_acc = tune_hyperparameters(
            X_train_tune, y_train_tune, classifier_name, cv_folds=5, use_linear_svm=use_linear_svm
        )
        
        # Train final model on full training set (or sample for SVM)
        best_model.fit(X_train_tune, y_train_tune)
        
        # Evaluate on train and test sets
        y_train_pred = best_model.predict(X_train_tune)
        y_test_pred = best_model.predict(X_test)
        
        train_acc = accuracy_score(y_train_tune, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        elapsed = time.time() - start_time
        
        # Sanity check for Mushroom dataset
        if dataset_name == 'mushroom' and test_acc < 0.95:
            print(f"WARNING: Mushroom test accuracy ({test_acc:.4f}) is below expected 95%+")
        
        return {
            'dataset': dataset_name,
            'classifier': classifier_name,
            'partition_ratio': partition_name,
            'trial': trial_seed,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'best_params': json.dumps(best_params),
            'time_elapsed': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR in {dataset_name}/{classifier_name}/{partition_name}/trial_{trial_seed}: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_all_experiments(output_file=None):
    """
    Run all experiments: 3 datasets × 3 partitions × 3 trials × 3 classifiers.
    
    Args:
        output_file: Path to save results CSV (default: results/experiment_results.csv)
    """
    if output_file is None:
        output_file = os.path.join(project_root, 'results', 'experiment_results.csv')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    datasets = ['adult', 'spambase', 'mushroom']
    classifiers = ['random_forest', 'svm', 'knn']
    partitions = get_partition_ratios()
    trial_seeds = get_trial_seeds()
    
    # For minimum requirements: use 50/50 partition with 1 trial
    # For comprehensive: use all partitions and trials
    # Set MINIMUM_MODE = True to run just 9 experiments (3×3)
    MINIMUM_MODE = False  # Set to True for minimum requirements only
    
    if MINIMUM_MODE:
        partitions = [('50_50', 0.5)]  # Just one partition
        trial_seeds = [42]  # Just one trial
        print("Running in MINIMUM MODE: 3 classifiers × 3 datasets = 9 experiments")
    else:
        print("Running in COMPREHENSIVE MODE: 3×3×3×3 = 81 experiments")
    
    results = []
    total_experiments = len(datasets) * len(classifiers) * len(partitions) * len(trial_seeds)
    
    # Load existing results if file exists
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict('records')
            print(f"Found {len(existing_results)} existing results. Continuing from there...")
            # Parse best_params JSON strings
            for r in existing_results:
                if isinstance(r.get('best_params'), str):
                    try:
                        r['best_params'] = json.loads(r['best_params'])
                    except:
                        pass
            results = existing_results
        except Exception as e:
            print(f"Could not load existing results: {e}. Starting fresh.")
    
    print(f"Running {total_experiments} experiments...")
    print("=" * 80)
    
    # Use tqdm for progress bar if available
    try:
        pbar = tqdm(total=total_experiments, initial=len(results), desc="Experiments")
        use_tqdm = True
    except:
        use_tqdm = False
        pbar = None
    
    completed_keys = set()
    if results:
        for r in results:
            key = (r['dataset'], r['classifier'], r['partition_ratio'], r['trial'])
            completed_keys.add(key)
    
    for dataset_name in datasets:
        for classifier_name in classifiers:
            for partition_name, train_ratio in partitions:
                for trial_seed in trial_seeds:
                    # Skip if already completed
                    key = (dataset_name, classifier_name, partition_name, trial_seed)
                    if key in completed_keys:
                        if use_tqdm:
                            pbar.update(1)
                        continue
                    
                    try:
                        print(f"\nStarting: {dataset_name}/{classifier_name}/{partition_name}/trial_{trial_seed}")
                        result = run_single_experiment(
                            dataset_name, classifier_name, 
                            partition_name, train_ratio, trial_seed
                        )
                        results.append(result)
                        completed_keys.add(key)
                        
                        # Save incrementally after each experiment
                        df_temp = pd.DataFrame(results)
                        df_temp.to_csv(output_file, index=False)
                        
                        if use_tqdm:
                            pbar.set_postfix({
                                'dataset': dataset_name,
                                'classifier': classifier_name,
                                'partition': partition_name,
                                'trial': trial_seed,
                                'test_acc': f"{result['test_acc']:.4f}",
                                'time': f"{result.get('time_elapsed', 0):.1f}s"
                            })
                        else:
                            print(f"✓ Completed: {dataset_name}/{classifier_name}/{partition_name}/trial_{trial_seed} - Test Acc: {result['test_acc']:.4f} ({result.get('time_elapsed', 0):.1f}s)")
                    except Exception as e:
                        print(f"\n✗ ERROR in {dataset_name}/{classifier_name}/{partition_name}/trial_{trial_seed}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next experiment
                    
                    if use_tqdm:
                        pbar.update(1)
    
    if use_tqdm:
        pbar.close()
    
    # Final save
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total experiments completed: {len(results)}/{total_experiments}")
    
    return df_results

if __name__ == '__main__':
    # Ensure results directory exists
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run all experiments
    results_df = run_all_experiments()
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"\nAverage test accuracies by classifier:")
    print(results_df.groupby('classifier')['test_acc'].mean().sort_values(ascending=False))
    print(f"\nAverage test accuracies by dataset:")
    print(results_df.groupby('dataset')['test_acc'].mean().sort_values(ascending=False))
