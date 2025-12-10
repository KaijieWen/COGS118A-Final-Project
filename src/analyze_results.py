"""
Analyze experiment results and generate tables and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(results_file=None):
    """
    Load experiment results from CSV.
    
    Args:
        results_file: Path to results CSV (default: results/experiment_results.csv)
        
    Returns:
        DataFrame with results
    """
    if results_file is None:
        results_file = os.path.join(project_root, 'results', 'experiment_results.csv')
    
    df = pd.read_csv(results_file)
    
    # Parse best_params JSON strings (handle both JSON strings and Python dict strings)
    def parse_params(param_str):
        if pd.isna(param_str):
            return {}
        if isinstance(param_str, dict):
            return param_str
        try:
            return json.loads(param_str)
        except:
            # Try eval for Python dict strings (with single quotes)
            try:
                import ast
                return ast.literal_eval(param_str)
            except:
                return {}
    
    df['best_params'] = df['best_params'].apply(parse_params)
    
    return df

def generate_summary_table(df):
    """
    Generate Table A: Average Test Accuracy (with std dev) for each combination.
    
    Args:
        df: Results DataFrame
        
    Returns:
        Summary table DataFrame
    """
    summary = df.groupby(['dataset', 'classifier', 'partition_ratio'])['test_acc'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    summary.columns = ['dataset', 'classifier', 'partition_ratio', 'mean_test_acc', 'std_test_acc', 'n_trials']
    summary = summary.sort_values(['dataset', 'classifier', 'partition_ratio'])
    
    return summary

def generate_hyperparameter_table(df):
    """
    Generate Table B: Best hyperparameters found (sample: Adult vs Spambase).
    
    Args:
        df: Results DataFrame
        
    Returns:
        Hyperparameter summary DataFrame
    """
    # Get one example of best params for each dataset-classifier combination
    # Use the first trial from 50/50 partition
    hp_data = []
    
    for dataset in ['adult', 'spambase']:
        for classifier in ['random_forest', 'svm', 'knn']:
            subset = df[(df['dataset'] == dataset) & 
                        (df['classifier'] == classifier) & 
                        (df['partition_ratio'] == '50_50')]
            
            if len(subset) > 0:
                # Get the first trial's params
                best_params = subset.iloc[0]['best_params']
                hp_data.append({
                    'dataset': dataset,
                    'classifier': classifier,
                    'best_params': json.dumps(best_params, indent=2)
                })
    
    hp_df = pd.DataFrame(hp_data)
    return hp_df

def plot_training_size_effect(df, output_file=None):
    """
    Plot A: Effect of Training Size (Line plot).
    X-axis: % Training Data (20%, 50%, 80%)
    Y-axis: Test Accuracy
    Separate lines for each classifier, grouped by dataset.
    
    Args:
        df: Results DataFrame
        output_file: Path to save plot
    """
    if output_file is None:
        output_file = os.path.join(project_root, 'results', 'training_size_effect.png')
    
    # Map partition ratios to percentages
    partition_map = {'20_80': 20, '50_50': 50, '80_20': 80}
    df['train_pct'] = df['partition_ratio'].map(partition_map)
    
    # Calculate mean and std for each combination
    plot_data = df.groupby(['dataset', 'classifier', 'train_pct'])['test_acc'].agg([
        'mean', 'std'
    ]).reset_index()
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    datasets = ['adult', 'spambase', 'mushroom']
    classifiers = ['random_forest', 'svm', 'knn']
    colors = {'random_forest': '#1f77b4', 'svm': '#ff7f0e', 'knn': '#2ca02c'}
    markers = {'random_forest': 'o', 'svm': 's', 'knn': '^'}
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = plot_data[plot_data['dataset'] == dataset]
        
        for clf in classifiers:
            clf_data = dataset_data[dataset_data['classifier'] == clf].sort_values('train_pct')
            if len(clf_data) > 0:
                ax.errorbar(
                    clf_data['train_pct'], 
                    clf_data['mean'],
                    yerr=clf_data['std'],
                    label=clf.replace('_', ' ').title(),
                    color=colors[clf],
                    marker=markers[clf],
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                    capthick=2
                )
        
        ax.set_xlabel('Training Data Percentage (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{dataset.title()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks([20, 50, 80])
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved training size effect plot to {output_file}")
    plt.close()

def plot_classifier_comparison(df, output_file=None):
    """
    Plot B: Classifier Comparison (Bar chart).
    Compare RF vs SVM vs KNN across all datasets.
    Group by partition ratio, show error bars (std dev).
    
    Args:
        df: Results DataFrame
        output_file: Path to save plot
    """
    if output_file is None:
        output_file = os.path.join(project_root, 'results', 'classifier_comparison.png')
    
    # Calculate mean and std for each combination
    plot_data = df.groupby(['dataset', 'classifier', 'partition_ratio'])['test_acc'].agg([
        'mean', 'std'
    ]).reset_index()
    
    # Create grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    datasets = ['adult', 'spambase', 'mushroom']
    classifiers = ['random_forest', 'svm', 'knn']
    partitions = ['20_80', '50_50', '80_20']
    partition_labels = ['20/80', '50/50', '80/20']
    colors = {'random_forest': '#1f77b4', 'svm': '#ff7f0e', 'knn': '#2ca02c'}
    
    x = np.arange(len(partitions))
    width = 0.25
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = plot_data[plot_data['dataset'] == dataset]
        
        for i, clf in enumerate(classifiers):
            clf_data = dataset_data[dataset_data['classifier'] == clf]
            means = []
            stds = []
            for part in partitions:
                part_data = clf_data[clf_data['partition_ratio'] == part]
                if len(part_data) > 0:
                    means.append(part_data['mean'].values[0])
                    stds.append(part_data['std'].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i*width, means, width, label=clf.replace('_', ' ').title(),
                   color=colors[clf], yerr=stds, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Train/Test Split', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{dataset.title()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(partition_labels)
        ax.set_ylim([0.5, 1.0])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved classifier comparison plot to {output_file}")
    plt.close()

def plot_accuracy_heatmap(df, output_file=None):
    """
    Heatmap of average test accuracy (classifier × dataset × partition).
    
    Args:
        df: Results DataFrame
        output_file: Path to save plot
    """
    if output_file is None:
        output_file = os.path.join(project_root, 'results', 'accuracy_heatmap.png')
    
    # Calculate mean test accuracy for each combination
    summary = df.groupby(['dataset', 'classifier', 'partition_ratio'])['test_acc'].mean().reset_index()
    
    # Create pivot table for heatmap
    # Format: classifier-dataset as rows, partition as columns
    heatmap_data = []
    for clf in ['random_forest', 'svm', 'knn']:
        for dataset in ['adult', 'spambase', 'mushroom']:
            row = {'classifier_dataset': f"{clf.replace('_', ' ').title()}\n({dataset})"}
            for part in ['20_80', '50_50', '80_20']:
                subset = summary[(summary['classifier'] == clf) & 
                                (summary['dataset'] == dataset) & 
                                (summary['partition_ratio'] == part)]
                if len(subset) > 0:
                    row[part] = subset['test_acc'].values[0]
                else:
                    row[part] = np.nan
            heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('classifier_dataset')
    heatmap_df.columns = ['20/80', '50/50', '80/20']
    
    # Create heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Test Accuracy'}, vmin=0.5, vmax=1.0)
    plt.title('Average Test Accuracy Heatmap\n(Classifier × Dataset × Partition)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Train/Test Split Ratio', fontsize=12)
    plt.ylabel('Classifier (Dataset)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy heatmap to {output_file}")
    plt.close()

def export_tables_to_latex(summary_df, hp_df, output_file=None):
    """
    Export summary tables to LaTeX format.
    
    Args:
        summary_df: Summary table DataFrame
        hp_df: Hyperparameter table DataFrame
        output_file: Path to save LaTeX file
    """
    if output_file is None:
        output_file = os.path.join(project_root, 'report', 'summary_tables.tex')
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Average Test Accuracy (with Standard Deviation) by Dataset, Classifier, and Partition Ratio}\n")
        f.write("\\label{tab:summary}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Classifier & Partition & Mean Acc (Std) \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in summary_df.iterrows():
            dataset = row['dataset'].title()
            classifier = row['classifier'].replace('_', ' ').title()
            partition = row['partition_ratio'].replace('_', '/')
            mean_acc = f"{row['mean_test_acc']:.4f}"
            std_acc = f"{row['std_test_acc']:.4f}"
            f.write(f"{dataset} & {classifier} & {partition} & {mean_acc} ({std_acc}) \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}%\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX tables to {output_file}")

def main():
    """Main analysis function."""
    print("Loading results...")
    df = load_results()
    
    print(f"Loaded {len(df)} experiment results")
    
    print("\nGenerating summary tables...")
    summary_df = generate_summary_table(df)
    summary_df.to_csv(os.path.join(project_root, 'results', 'summary_tables.csv'), index=False)
    print("Summary table saved to results/summary_tables.csv")
    
    hp_df = generate_hyperparameter_table(df)
    hp_df.to_csv(os.path.join(project_root, 'results', 'best_hyperparameters.csv'), index=False)
    print("Hyperparameter table saved to results/best_hyperparameters.csv")
    
    print("\nGenerating visualizations...")
    plot_training_size_effect(df)
    plot_classifier_comparison(df)
    plot_accuracy_heatmap(df)
    
    # Copy figures to report/figures directory
    import shutil
    figures_dir = os.path.join(project_root, 'report', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    figure_files = [
        ('training_size_effect.png', 'training_size_effect.png'),
        ('classifier_comparison.png', 'classifier_comparison.png'),
        ('accuracy_heatmap.png', 'accuracy_heatmap.png')
    ]
    
    for src_name, dst_name in figure_files:
        src = os.path.join(project_root, 'results', src_name)
        dst = os.path.join(figures_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src_name} to report/figures/")
    
    print("\nExporting LaTeX tables...")
    export_tables_to_latex(summary_df, hp_df)
    
    print("\nAnalysis complete!")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("\nAverage test accuracy by classifier:")
    print(df.groupby('classifier')['test_acc'].agg(['mean', 'std']).sort_values('mean', ascending=False))
    print("\nAverage test accuracy by dataset:")
    print(df.groupby('dataset')['test_acc'].agg(['mean', 'std']).sort_values('mean', ascending=False))
    print("\nAverage test accuracy by partition ratio:")
    print(df.groupby('partition_ratio')['test_acc'].agg(['mean', 'std']).sort_values('mean', ascending=False))

if __name__ == '__main__':
    main()

