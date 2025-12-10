# Machine Learning Final Project: Empirical Classification Study

## Project Overview

This project replicates and extends the empirical comparison of classification algorithms conducted by Caruana and Niculescu-Mizil. We evaluate three classifiers (Random Forest, SVM, KNN) on three binary classification datasets from the UCI repository, following a comprehensive experimental design.

## Project Structure

```
COGS118A_Final_Project/
├── data/                    # Dataset files
│   ├── adult/              # Adult (Census Income) dataset
│   ├── spambase/           # Spambase dataset
│   └── mushroom/           # Mushroom dataset
├── src/                     # Python source code
│   ├── load_adult.py       # Adult dataset loader
│   ├── load_mushroom.py    # Mushroom dataset loader
│   ├── load_spambase.py    # Spambase dataset loader
│   ├── data_splits.py      # Data splitting utilities
│   ├── hyperparameter_tuning.py  # Hyperparameter tuning
│   ├── run_experiments.py  # Master experiment loop
│   └── analyze_results.py # Results analysis and visualization
├── results/                 # Experimental results
│   ├── experiment_results.csv
│   ├── summary_tables.csv
│   └── *.png               # Visualization plots
├── report/                  # LaTeX report
│   ├── main.tex
│   ├── neurips_2023.sty
│   └── figures/
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Execute all 81 experiments (3 datasets × 3 classifiers × 3 partitions × 3 trials):
```bash
python src/run_experiments.py
```

This will create `results/experiment_results.csv` with all experimental results.

**Note:** This process takes several hours to complete due to extensive hyperparameter tuning with 5-fold cross-validation.

### Analyzing Results

After experiments complete, generate summary tables and visualizations:
```bash
python src/analyze_results.py
```

Or use the automated checker:
```bash
python check_and_analyze.py
```

This generates:
- Summary tables (CSV and LaTeX)
- Training size effect plots
- Classifier comparison plots
- Accuracy heatmaps

### Compiling Report

To compile the LaTeX report:
```bash
cd report
pdflatex main.tex
bibtex main  # if using bibliography
pdflatex main.tex
pdflatex main.tex
```

## Experimental Design

### Datasets
1. **Adult (Census Income)**: 48,842 instances, mixed categorical/numerical features
2. **Spambase**: 4,601 instances, 57 continuous features
3. **Mushroom**: 8,124 instances, 22 categorical features

### Classifiers
1. **Random Forest**: Tree-based ensemble
2. **Support Vector Machine (SVM)**: Kernel-based
3. **K-Nearest Neighbors (KNN)**: Instance-based

### Experimental Matrix
- 3 datasets
- 3 classifiers
- 3 partition ratios (20/80, 50/50, 80/20 train/test)
- 3 independent trials per combination
- **Total: 81 experiments**

### Hyperparameter Tuning
- Method: GridSearchCV with 5-fold cross-validation
- Performed exclusively on training set
- Best parameters selected based on CV accuracy

## Results

Results are saved in `results/experiment_results.csv` with the following columns:
- `dataset`: Dataset name
- `classifier`: Classifier name
- `partition_ratio`: Train/test split ratio
- `trial`: Random seed used
- `train_acc`: Training accuracy
- `val_acc`: Cross-validation accuracy (best CV score)
- `test_acc`: Test set accuracy
- `best_params`: Best hyperparameters found (JSON string)

## Key Findings

1. **Random Forest** consistently achieves the highest test accuracy across all datasets
2. **Training data size** has a significant positive effect on performance
3. **Dataset characteristics** strongly influence relative algorithm performance
4. Results align with original Caruana and Niculescu-Mizil findings

## References

- Caruana, R., & Niculescu-Mizil, A. (2006). An empirical comparison of supervised learning algorithms. ICML.
- UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/

## License

This project is for educational purposes as part of COGS 118A coursework.


