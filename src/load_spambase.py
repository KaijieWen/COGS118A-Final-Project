"""
Loader for Spambase dataset.
Standardizes features for SVM/KNN sensitivity.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_spambase_dataset(data_dir=None, standardize=True):
    """
    Load and preprocess the Spambase dataset.
    
    Args:
        data_dir: Path to the spambase dataset directory (default: data/spambase relative to project root)
        standardize: Whether to standardize features (default: True)
        
    Returns:
        X: Feature matrix (numpy array, standardized if standardize=True)
        y: Target vector (numpy array, binary: 1 for spam, 0 for non-spam)
        scaler: StandardScaler object (if standardize=True) or None
    """
    if data_dir is None:
        # Get project root (parent of src directory)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        data_dir = os.path.join(project_root, 'data', 'spambase')
    
    # Load data (comma-separated, last column is target)
    data_path = os.path.join(data_dir, 'spambase.data')
    df = pd.read_csv(data_path, header=None)
    
    # Last column is target (1=spam, 0=non-spam)
    y = df.iloc[:, -1].values.astype(int)
    
    # All other columns are features
    X = df.iloc[:, :-1].values.astype(float)
    
    # Standardize features (mean=0, std=1) for SVM/KNN
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    print(f"Spambase dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y==0)} (non-spam), {np.sum(y==1)} (spam)")
    if standardize:
        print(f"Features standardized: mean={X.mean():.4f}, std={X.std():.4f}")
    
    return X, y, scaler

if __name__ == '__main__':
    X, y, scaler = load_spambase_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

