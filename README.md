# Machine Learning Final Project: Empirical Classification Study

## Project Overview

This project replicates and extends the empirical comparison of classification algorithms conducted by Caruana and Niculescu-Mizil. We evaluate three classifiers (Random Forest, SVM, KNN) on three binary classification datasets from the UCI repository, following a comprehensive experimental design.

## Project Structure

```
COGS118A_Final_Project/
├── data/                          # Dataset files
│   ├── adult/                     # Adult (Census Income) dataset
│   │   ├── adult.data
│   │   ├── adult.names
│   │   ├── adult.test
│   │   └── Index                  # Dataset metadata
│   ├── mushroom/                  # Mushroom dataset
│   │   ├── agaricus-lepiota.data
│   │   ├── agaricus-lepiota.names
│   │   ├── expanded.Z
│   │   ├── Index
│   │   └── README
│   └── spambase/                  # Spambase dataset
│       ├── spambase.data
│       ├── spambase.names
│       └── spambase.DOCUMENTATION
├── src/                           # Python source code
│   ├── load_adult.py             # Adult dataset loader and preprocessor
│   ├── load_mushroom.py          # Mushroom dataset loader and preprocessor
│   ├── load_spambase.py          # Spambase dataset loader and preprocessor
│   ├── data_splits.py            # Data splitting utilities (StratifiedShuffleSplit)
│   ├── hyperparameter_tuning.py  # GridSearchCV hyperparameter tuning
│   ├── run_experiments.py        # Master experiment loop (81 experiments)
│   └── analyze_results.py        # Results analysis and visualization generator
├── results/                       # Experimental results and analysis outputs
│   ├── experiment_results.csv     # Complete experiment results (81 experiments)
│   ├── summary_tables.csv         # Aggregated summary statistics
│   ├── best_hyperparameters.csv   # Best hyperparameters found per dataset/classifier
│   ├── experiment_log.txt         # Experiment execution log
│   ├── accuracy_heatmap.png       # Generated visualization (heatmap)
│   ├── classifier_comparison.png  # Generated visualization (bar charts)
│   └── training_size_effect.png   # Generated visualization (line plots)
├── report/                        # Report visualizations
│   └── figures/                   # Visualization figures (copies for report)
│       ├── accuracy_heatmap.png
│       ├── classifier_comparison.png
│       └── training_size_effect.png
├── check_and_analyze.py          # Utility script to check experiment completion
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

**Note:** LaTeX report files (main.tex, neurips_2023.sty, summary_tables.tex) have been removed from this repository. The project focuses on experimental execution and results analysis. Visualization outputs are available as PNG files in the `results/` and `report/figures/` directories.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:KaijieWen/COGS118A-Final-Project.git
cd COGS118A-Final-Project
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0 (optional, for progress bars)

## Usage

### Running Experiments

Execute all 81 experiments (3 datasets × 3 classifiers × 3 partitions × 3 trials):
```bash
python src/run_experiments.py
```

This script will:
- Load each dataset with appropriate preprocessing
- Split data using stratified sampling for three partition ratios (20/80, 50/50, 80/20)
- Perform hyperparameter tuning using 5-fold cross-validation (3-fold for SVM)
- Train final models and evaluate on test sets
- Save results incrementally to `results/experiment_results.csv`

**Note:** This process takes several hours to complete due to extensive hyperparameter tuning. The script saves results incrementally, so you can stop and resume if needed. Progress is shown with progress bars if `tqdm` is installed.

**Experiment Matrix:**
- **Datasets:** Adult, Spambase, Mushroom (3 total)
- **Classifiers:** Random Forest, SVM, KNN (3 total)
- **Partition Ratios:** 20/80, 50/50, 80/20 train/test (3 total)
- **Trials:** 3 independent runs with different random seeds (42, 123, 456)
- **Total:** 3 × 3 × 3 × 3 = 81 experiments

### Analyzing Results

After experiments complete (or to re-analyze existing results), generate summary tables and visualizations:
```bash
python src/analyze_results.py
```

This will generate:
- **Summary tables:** `results/summary_tables.csv` (mean accuracy with std dev)
- **Hyperparameter table:** `results/best_hyperparameters.csv` (best params per dataset/classifier)
- **Visualizations:** 
  - `results/training_size_effect.png` - Effect of training data size
  - `results/classifier_comparison.png` - Classifier comparison bar charts
  - `results/accuracy_heatmap.png` - Comprehensive accuracy heatmap
- **LaTeX table export:** `report/summary_tables.tex` (for potential report integration)
- **Figure copies:** All visualizations copied to `report/figures/` for easy access

**Alternative:** Use the automated checker script:
```bash
python check_and_analyze.py
```

This script checks if all 81 experiments are complete, and if so, automatically runs the analysis.

## Experimental Design

### Datasets

1. **Adult (Census Income)**
   - **Size:** 48,842 instances, 14 features
   - **Type:** Mixed categorical and numerical features
   - **Task:** Binary classification (income >$50K vs ≤$50K)
   - **Preprocessing:** Missing value imputation, one-hot encoding for categorical features (105 features after encoding)
   - **Class distribution:** ~76% ≤50K, ~24% >50K

2. **Spambase**
   - **Size:** 4,601 instances, 57 features
   - **Type:** All continuous features (word/character frequencies, capital letter statistics)
   - **Task:** Binary classification (spam vs non-spam)
   - **Preprocessing:** Standardization (mean=0, std=1) using StandardScaler
   - **Class distribution:** Balanced

3. **Mushroom**
   - **Size:** 8,124 instances, 22 features
   - **Type:** All categorical features
   - **Task:** Binary classification (poisonous vs edible)
   - **Preprocessing:** One-hot encoding for all categorical features
   - **Class distribution:** ~52% edible, ~48% poisonous

### Classifiers

1. **Random Forest**
   - Implementation: `sklearn.ensemble.RandomForestClassifier`
   - Hyperparameters tuned: `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {10, 20, None}, `min_samples_split` ∈ {2, 5, 10}
   - Total grid size: 3 × 3 × 3 = 27 combinations

2. **Support Vector Machine (SVM)**
   - Implementation: `sklearn.svm.SVC` (or `LinearSVC` for large datasets)
   - Hyperparameters tuned: `C` ∈ {0.1, 1, 10}, `kernel` ∈ {linear, rbf}, `gamma` ∈ {'scale', 0.001} (for RBF only)
   - Optimization: Uses LinearSVC with sampling for datasets >5000 samples to reduce computation time
   - Grid structure: Separate grids for linear (3 C values) and RBF (3 C × 2 gamma = 6 combinations)
   - Total combinations: 3 (linear) + 6 (RBF) = 9 combinations

3. **K-Nearest Neighbors (KNN)**
   - Implementation: `sklearn.neighbors.KNeighborsClassifier`
   - Hyperparameters tuned: `n_neighbors` ∈ {3, 5, 7, 9, 11}, `weights` ∈ {uniform, distance}
   - Total grid size: 5 × 2 = 10 combinations

### Hyperparameter Tuning

- **Method:** GridSearchCV with exhaustive grid search
- **Cross-validation:** 5-fold CV for RF and KNN, 3-fold CV for SVM (to reduce computation time)
- **Scoring metric:** Accuracy
- **Scope:** Tuning performed exclusively on training set (no test set leakage)
- **Selection:** Best hyperparameters selected based on cross-validation accuracy

### Evaluation Protocol

- **Data splitting:** StratifiedShuffleSplit to maintain class balance across train/test splits
- **Metrics reported:**
  - **Train Accuracy:** Accuracy on full training set
  - **Validation Accuracy:** Best cross-validation score from hyperparameter tuning
  - **Test Accuracy:** Accuracy on held-out test set (primary metric)
- **Statistical robustness:** 3 independent trials per condition with different random seeds
- **Results aggregation:** Mean and standard deviation calculated across trials

## Results

### Output Files

All results are saved in `results/` directory:

1. **`experiment_results.csv`**
   - Complete results for all 81 experiments
   - Columns: `dataset`, `classifier`, `partition_ratio`, `trial`, `train_acc`, `val_acc`, `test_acc`, `best_params`, `time_elapsed`

2. **`summary_tables.csv`**
   - Aggregated statistics (mean and std dev) by dataset, classifier, and partition ratio
   - Generated by `analyze_results.py`

3. **`best_hyperparameters.csv`**
   - Best hyperparameters found for each dataset-classifier combination (50/50 partition, trial 1)
   - Useful for understanding optimal configurations

4. **Visualizations** (PNG format, ~150-280 KB each):
   - `training_size_effect.png`: Line plots showing accuracy vs. training data percentage
   - `classifier_comparison.png`: Bar charts comparing classifiers across datasets and partitions
   - `accuracy_heatmap.png`: Comprehensive heatmap of all experimental conditions

### Key Findings

Based on the experimental results:

1. **Random Forest** consistently achieves the highest test accuracy across all datasets and partition ratios
   - Average accuracies: ~85% (Adult), ~93% (Spambase), ~99.8% (Mushroom)

2. **Training data size** has a significant positive effect on performance
   - Improvement from 20% to 80% training data: ~5% for Adult, ~2-3% for Spambase
   - Mushroom achieves near-perfect accuracy even with 20% training data

3. **Dataset characteristics** strongly influence relative algorithm performance
   - Mushroom: Near-perfect separability allows all classifiers to excel
   - Adult: Mixed feature types and class imbalance present challenges
   - Spambase: Continuous features with standardization benefit SVM

4. **SVM performance** varies significantly with dataset characteristics
   - Competitive on Spambase (continuous features)
   - Struggles on Adult (high-dimensional one-hot encoded categorical features)
   - Optimization with LinearSVC improves efficiency for large datasets

5. **KNN performance** reflects curse of dimensionality
   - Excellent on Mushroom (categorical, well-separated)
   - Struggles on high-dimensional Adult dataset
   - Competitive baseline overall

These findings align with the original Caruana and Niculescu-Mizil study conclusions regarding algorithm performance and the importance of training data size.

## Source Code Overview

### Data Loading Modules (`src/load_*.py`)

Each dataset has a dedicated loader that handles:
- File I/O and parsing
- Missing value imputation
- Feature encoding (one-hot for categorical, standardization for continuous)
- Target variable encoding (binary: 0/1)
- Data shape and distribution reporting

**Functions:**
- `load_adult_dataset(data_dir=None)` → X, y
- `load_mushroom_dataset(data_dir=None)` → X, y
- `load_spambase_dataset(data_dir=None, standardize=True)` → X, y, scaler

### Data Splitting (`src/data_splits.py`)

**Functions:**
- `get_train_test_split(X, y, train_ratio, random_state=42)` → X_train, X_test, y_train, y_test
- `get_partition_ratios()` → List of (name, ratio) tuples
- `get_trial_seeds()` → List of random seeds [42, 123, 456]

### Hyperparameter Tuning (`src/hyperparameter_tuning.py`)

**Functions:**
- `get_hyperparameter_grids()` → Dictionary of parameter grids
- `get_classifier(classifier_name, use_linear_svm=False)` → Classifier instance
- `tune_hyperparameters(X_train, y_train, classifier_name, cv_folds=5, n_jobs=-1, use_linear_svm=False)` → best_model, best_params, best_cv_score

### Experiment Execution (`src/run_experiments.py`)

**Main functions:**
- `load_dataset(dataset_name)` → X, y
- `run_single_experiment(...)` → Result dictionary
- `run_all_experiments(output_file=None)` → Results DataFrame

**Features:**
- Incremental result saving (can resume after interruption)
- Progress tracking with tqdm (if available)
- Error handling and logging
- SVM optimization for large datasets

### Results Analysis (`src/analyze_results.py`)

**Functions:**
- `load_results(results_file=None)` → Results DataFrame
- `generate_summary_table(df)` → Summary DataFrame
- `generate_hyperparameter_table(df)` → Hyperparameter DataFrame
- `plot_training_size_effect(df, output_file=None)` → Saves PNG
- `plot_classifier_comparison(df, output_file=None)` → Saves PNG
- `plot_accuracy_heatmap(df, output_file=None)` → Saves PNG
- `export_tables_to_latex(summary_df, hp_df, output_file=None)` → Saves .tex file
- `main()` → Complete analysis pipeline

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Trial seeds: 42, 123, 456
- Classifier random states: 42 (for RF and SVM)
- Data splitting: Controlled via `random_state` parameter

Results can be reproduced by running:
```bash
python src/run_experiments.py
```

## References

- Caruana, R., & Niculescu-Mizil, A. (2006). An empirical comparison of supervised learning algorithms. *Proceedings of the 23rd international conference on Machine learning*, 161-168.
- Caruana, R., Karampatziakis, N., & Yessenalina, A. (2008). An empirical evaluation of supervised learning in high dimensions. *Proceedings of the 25th international conference on Machine learning*, 96-103.
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12(Oct), 2825-2830.

## License

This project is for educational purposes as part of COGS 118A coursework at UC San Diego.

## Repository

GitHub: https://github.com/KaijieWen/COGS118A-Final-Project
