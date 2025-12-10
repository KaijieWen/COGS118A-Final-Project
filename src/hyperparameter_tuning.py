"""
Hyperparameter tuning using GridSearchCV with 5-fold cross-validation.
Defines parameter grids for Random Forest, SVM, and KNN.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def get_hyperparameter_grids():
    """
    Define hyperparameter grids for each classifier.
    
    Returns:
        Dictionary mapping classifier names to parameter grids
    """
    grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'svm': {
            'C': [0.1, 1, 10],  # Reduced from 4 to 3 values
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 0.001]  # Further reduced to 2 values for speed
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    }
    return grids

def get_classifier(classifier_name, use_linear_svm=False):
    """
    Get a classifier instance by name.
    
    Args:
        classifier_name: 'random_forest', 'svm', or 'knn'
        use_linear_svm: If True, use LinearSVC instead of SVC (faster for large datasets)
        
    Returns:
        Classifier instance
    """
    if classifier_name == 'svm' and use_linear_svm:
        return LinearSVC(random_state=42, max_iter=1000, dual=False)
    classifiers = {
        'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'svm': SVC(random_state=42),
        'knn': KNeighborsClassifier()
    }
    return classifiers[classifier_name]

def tune_hyperparameters(X_train, y_train, classifier_name, cv_folds=5, n_jobs=-1, use_linear_svm=False):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        classifier_name: 'random_forest', 'svm', or 'knn'
        cv_folds: Number of CV folds (default: 5)
        n_jobs: Number of parallel jobs (default: -1 for all cores)
        use_linear_svm: If True, use LinearSVC instead of SVC (faster for large datasets)
        
    Returns:
        best_model: Best model fitted on full training data
        best_params: Best hyperparameters found
        best_cv_score: Best cross-validation score
    """
    grids = get_hyperparameter_grids()
    
    # Use LinearSVC for large datasets
    if classifier_name == 'svm' and use_linear_svm:
        classifier = get_classifier(classifier_name, use_linear_svm=True)
        # LinearSVC only has C parameter
        param_grid = {'C': grids['svm']['C']}
    else:
        classifier = get_classifier(classifier_name)
        param_grid = grids[classifier_name]
        
        # Handle SVM gamma parameter (only for RBF kernel)
        if classifier_name == 'svm':
            # Create separate grids for linear and RBF kernels
            linear_grid = {'C': param_grid['C'], 'kernel': ['linear']}
            rbf_grid = {
                'C': param_grid['C'],
                'kernel': ['rbf'],
                'gamma': param_grid['gamma']
            }
            # Use list of dicts for GridSearchCV
            param_grid = [linear_grid, rbf_grid]
    
    # Perform grid search with 5-fold CV
    # Reduce n_jobs for SVM to avoid memory issues
    # Use 3-fold CV for SVM to speed up significantly
    actual_n_jobs = 1 if classifier_name == 'svm' else n_jobs
    actual_cv = 3 if classifier_name == 'svm' else cv_folds
    
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=actual_cv,
        scoring='accuracy',
        n_jobs=actual_n_jobs,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    return best_model, best_params, best_cv_score

if __name__ == '__main__':
    # Test hyperparameter tuning
    from load_spambase import load_spambase_dataset
    from data_splits import get_train_test_split
    
    X, y, _ = load_spambase_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, 0.8, random_state=42)
    
    print("Testing hyperparameter tuning...")
    for clf_name in ['random_forest', 'svm', 'knn']:
        print(f"\nTuning {clf_name}...")
        best_model, best_params, best_cv_score = tune_hyperparameters(
            X_train, y_train, clf_name, cv_folds=5
        )
        print(f"Best CV score: {best_cv_score:.4f}")
        print(f"Best params: {best_params}")

