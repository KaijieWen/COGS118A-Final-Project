"""
Loader for Adult (Census Income) dataset.
Handles missing values, categorical encoding, and target conversion.
"""

import pandas as pd
import numpy as np
import os

def load_adult_dataset(data_dir=None):
    """
    Load and preprocess the Adult dataset.
    
    Args:
        data_dir: Path to the adult dataset directory (default: data/adult relative to project root)
        
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array, binary: 1 for >50K, 0 for <=50K)
    """
    if data_dir is None:
        # Get project root (parent of src directory)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        data_dir = os.path.join(project_root, 'data', 'adult')
    
    # Column names based on adult.names
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Load training and test data
    train_path = os.path.join(data_dir, 'adult.data')
    test_path = os.path.join(data_dir, 'adult.test')
    
    # Read data files (they have trailing spaces and commas)
    train_df = pd.read_csv(train_path, names=columns, sep=r',\s*', engine='python', na_values='?')
    test_df = pd.read_csv(test_path, names=columns, sep=r',\s*', engine='python', na_values='?', skiprows=1)
    
    # Remove trailing periods from test set target values
    test_df['income'] = test_df['income'].str.rstrip('.')
    
    # Combine train and test for full dataset
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Separate features and target
    target_col = 'income'
    y = df[target_col].copy()
    
    # Encode target: >50K -> 1, <=50K -> 0
    y = (y == '>50K').astype(int)
    
    # Drop target from features
    X_df = df.drop(columns=[target_col])
    
    # Handle missing values
    # For categorical: replace with mode
    # For numerical: replace with median
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                     'capital-loss', 'hours-per-week']
    
    for col in categorical_cols:
        if col in X_df.columns:
            mode_val = X_df[col].mode()[0] if not X_df[col].mode().empty else 'Unknown'
            X_df[col] = X_df[col].fillna(mode_val)
    
    for col in numerical_cols:
        if col in X_df.columns:
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)
    
    # One-Hot Encoding for categorical columns
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, prefix=categorical_cols)
    
    # Ensure numerical columns are included
    for col in numerical_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = X_df[col]
    
    # Reorder columns: numerical first, then encoded categorical
    numerical_cols_present = [col for col in numerical_cols if col in X_encoded.columns]
    categorical_cols_encoded = [col for col in X_encoded.columns if col not in numerical_cols_present]
    X_encoded = X_encoded[numerical_cols_present + categorical_cols_encoded]
    
    # Convert to numpy arrays
    X = X_encoded.values.astype(float)
    y = y.values
    
    print(f"Adult dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y==0)} (<=50K), {np.sum(y==1)} (>50K)")
    
    return X, y

if __name__ == '__main__':
    X, y = load_adult_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

