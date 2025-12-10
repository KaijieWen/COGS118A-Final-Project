"""
Loader for Mushroom dataset.
Handles missing values and applies one-hot encoding to all categorical features.
"""

import pandas as pd
import numpy as np
import os

def load_mushroom_dataset(data_dir=None):
    """
    Load and preprocess the Mushroom dataset.
    
    Args:
        data_dir: Path to the mushroom dataset directory (default: data/mushroom relative to project root)
        
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array, binary: 1 for poisonous, 0 for edible)
    """
    if data_dir is None:
        # Get project root (parent of src directory)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        data_dir = os.path.join(project_root, 'data', 'mushroom')
    
    # Column names based on agaricus-lepiota.names
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    # Load data
    data_path = os.path.join(data_dir, 'agaricus-lepiota.data')
    df = pd.read_csv(data_path, names=columns, na_values='?')
    
    # Separate target (first column: 'p' for poisonous, 'e' for edible)
    target_col = 'class'
    y = df[target_col].copy()
    
    # Encode target: p (poisonous) -> 1, e (edible) -> 0
    y = (y == 'p').astype(int)
    
    # Drop target from features
    X_df = df.drop(columns=[target_col])
    
    # Handle missing values in stalk-root (attribute #11)
    # Replace '?' with 'missing' category
    if 'stalk-root' in X_df.columns:
        X_df['stalk-root'] = X_df['stalk-root'].fillna('missing')
        # Also handle any other missing values
        for col in X_df.columns:
            if X_df[col].isna().any():
                mode_val = X_df[col].mode()[0] if not X_df[col].mode().empty else 'missing'
                X_df[col] = X_df[col].fillna(mode_val)
    
    # One-Hot Encoding for all categorical features
    X_encoded = pd.get_dummies(X_df, prefix=X_df.columns)
    
    # Convert to numpy arrays
    X = X_encoded.values.astype(float)
    y = y.values
    
    print(f"Mushroom dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y==0)} (edible), {np.sum(y==1)} (poisonous)")
    
    return X, y

if __name__ == '__main__':
    X, y = load_mushroom_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

