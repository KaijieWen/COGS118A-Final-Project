"""
Data splitting utilities using StratifiedShuffleSplit.
Implements 20/80, 50/50, and 80/20 train/test splits.
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def get_train_test_split(X, y, train_ratio, random_state=42):
    """
    Split data into train and test sets using StratifiedShuffleSplit.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_ratio: Proportion of data for training (e.g., 0.2 for 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def get_partition_ratios():
    """
    Get the three partition ratios: 20/80, 50/50, 80/20.
    
    Returns:
        List of tuples: [(ratio_name, train_ratio), ...]
    """
    return [
        ('20_80', 0.2),
        ('50_50', 0.5),
        ('80_20', 0.8)
    ]

def get_trial_seeds():
    """
    Get the three random seeds for trials.
    
    Returns:
        List of seeds: [42, 123, 456]
    """
    return [42, 123, 456]

if __name__ == '__main__':
    # Test the splitting function
    from load_adult import load_adult_dataset
    
    X, y = load_adult_dataset()
    print(f"Full dataset: {X.shape[0]} samples")
    
    for ratio_name, train_ratio in get_partition_ratios():
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, train_ratio, random_state=42)
        print(f"{ratio_name}: Train={X_train.shape[0]}, Test={X_test.shape[0]}")


