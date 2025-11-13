"""
Evaluation and visualization script.
Generates plots and analysis from experimental results.
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results():
    """
    Load experimental results.
    
    Returns:
        tuple: (DataFrame of metrics, list of detailed results)
    """
    # Load CSV metrics
    metrics_df = pd.read_csv('results/metrics.csv')
    
    # Load detailed results if available
    detailed_results = None
    if os.path.exists('results/detailed_results.pkl'):
        with open('results/detailed_results.pkl', 'rb') as f:
            detailed_results = pickle.load(f)
    
    return metrics_df, detailed_results


def plot_accuracy_vs_seq_length(metrics_df):
    """
    Plot accuracy vs sequence length.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    plt.figure(figsize=(10, 6))
    
    # Filter for consistent configuration (e.g., LSTM with Adam, no grad clipping)
    filtered_df = metrics_df[
        (metrics_df['Model'] == 'LSTM') & 
        (metrics_df['Optimizer'] == 'adam') & 
        (metrics_df['Grad Clipping'] == 'No')
    ]
    
    if len(filtered_df) > 0:
        seq_lengths = filtered_df['Seq Length'].values
        accuracies = filtered_df['Accuracy'].astype(float).values
        
        plt.plot(seq_lengths, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy vs Sequence Length (LSTM, Adam, No Grad Clipping)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(seq_lengths)
    
    plt.tight_layout()
    plt.savefig('results/plots/accuracy_vs_seq_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/accuracy_vs_seq_length.png")


def plot_f1_vs_seq_length(metrics_df):
    """
    Plot F1 score vs sequence length.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    plt.figure(figsize=(10, 6))
    
    # Filter for consistent configuration
    filtered_df = metrics_df[
        (metrics_df['Model'] == 'LSTM') & 
        (metrics_df['Optimizer'] == 'adam') & 
        (metrics_df['Grad Clipping'] == 'No')
    ]
    
    if len(filtered_df) > 0:
        seq_lengths = filtered_df['Seq Length'].values
        f1_scores = filtered_df['F1'].astype(float).values
        
        plt.plot(seq_lengths, f1_scores, marker='s', linewidth=2, markersize=8, color='green')
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score vs Sequence Length (LSTM, Adam, No Grad Clipping)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(seq_lengths)
    
    plt.tight_layout()
    plt.savefig('results/plots/f1_vs_seq_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/f1_vs_seq_length.png")


def plot_training_curves(detailed_results):
    """
    Plot training loss curves for best and worst models.
    
    Args:
        detailed_results (list): List of detailed result dictionaries
    """
    if detailed_results is None or len(detailed_results) == 0:
        print("No detailed results available for training curves")
        return
    
    # Find best and worst models by test accuracy
    accuracies = [r['test_accuracy'] for r in detailed_results]
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)
    
    best_model = detailed_results[best_idx]
    worst_model = detailed_results[worst_idx]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot best model
    epochs = range(1, len(best_model['train_losses']) + 1)
    axes[0].plot(epochs, best_model['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, best_model['test_losses'], 'r--', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Best Model: {best_model["model_type"].upper()}\n'
                     f'Acc: {best_model["test_accuracy"]:.2f}%, F1: {best_model["test_f1"]:.4f}',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot worst model
    epochs = range(1, len(worst_model['train_losses']) + 1)
    axes[1].plot(epochs, worst_model['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[1].plot(epochs, worst_model['test_losses'], 'r--', label='Test Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title(f'Worst Model: {worst_model["model_type"].upper()}\n'
                     f'Acc: {worst_model["test_accuracy"]:.2f}%, F1: {worst_model["test_f1"]:.4f}',
                     fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/training_curves_best_worst.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/training_curves_best_worst.png")


def plot_model_comparison(metrics_df):
    """
    Compare different model architectures.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    plt.figure(figsize=(12, 6))
    
    # Filter for baseline configurations
    filtered_df = metrics_df[
        (metrics_df['Activation'] == 'tanh') & 
        (metrics_df['Optimizer'] == 'adam') & 
        (metrics_df['Seq Length'] == 50) &
        (metrics_df['Grad Clipping'] == 'No')
    ]
    
    if len(filtered_df) > 0:
        models = filtered_df['Model'].values
        accuracies = filtered_df['Accuracy'].astype(float).values
        f1_scores = filtered_df['F1'].astype(float).values
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', alpha=0.8)
        plt.bar(x + width/2, f1_scores * 100, width, label='F1 Score (%)', alpha=0.8)
        
        plt.xlabel('Model Architecture', fontsize=12)
        plt.ylabel('Performance (%)', fontsize=12)
        plt.title('Model Architecture Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/model_comparison.png")


def plot_optimizer_comparison(metrics_df):
    """
    Compare different optimizers.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    plt.figure(figsize=(12, 6))
    
    # Filter for LSTM with tanh, seq length 50
    filtered_df = metrics_df[
        (metrics_df['Model'] == 'LSTM') & 
        (metrics_df['Activation'] == 'tanh') & 
        (metrics_df['Seq Length'] == 50) &
        (metrics_df['Grad Clipping'] == 'No')
    ]
    
    if len(filtered_df) > 0:
        optimizers = filtered_df['Optimizer'].values
        accuracies = filtered_df['Accuracy'].astype(float).values
        epoch_times = filtered_df['Epoch Time (s)'].astype(float).values
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(optimizers))
        width = 0.35
        
        # Plot accuracy on left y-axis
        ax1.bar(x, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Optimizer', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels([opt.upper() for opt in optimizers])
        
        # Plot epoch time on right y-axis
        ax2 = ax1.twinx()
        ax2.plot(x, epoch_times, 'ro-', linewidth=2, markersize=8, label='Epoch Time')
        ax2.set_ylabel('Epoch Time (s)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Optimizer Comparison (LSTM, Tanh, Seq Len 50)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/optimizer_comparison.png")


def plot_gradient_clipping_effect(metrics_df):
    """
    Show the effect of gradient clipping.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    plt.figure(figsize=(12, 6))
    
    # Compare gradient clipping for each model type
    model_types = ['RNN', 'LSTM', 'BILSTM']
    clip_no = []
    clip_yes = []
    
    for model in model_types:
        filtered_df = metrics_df[
            (metrics_df['Model'] == model) & 
            (metrics_df['Optimizer'] == 'adam') & 
            (metrics_df['Seq Length'] == 50)
        ]
        
        no_clip = filtered_df[filtered_df['Grad Clipping'] == 'No']
        yes_clip = filtered_df[filtered_df['Grad Clipping'] == 'Yes']
        
        if len(no_clip) > 0:
            clip_no.append(no_clip['Accuracy'].astype(float).values[0])
        else:
            clip_no.append(0)
        
        if len(yes_clip) > 0:
            clip_yes.append(yes_clip['Accuracy'].astype(float).values[0])
        else:
            clip_yes.append(0)
    
    x = np.arange(len(model_types))
    width = 0.35
    
    plt.bar(x - width/2, clip_no, width, label='No Gradient Clipping', alpha=0.8)
    plt.bar(x + width/2, clip_yes, width, label='With Gradient Clipping', alpha=0.8)
    
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Gradient Clipping', fontsize=14, fontweight='bold')
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/gradient_clipping_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/gradient_clipping_effect.png")


def plot_heatmap(metrics_df):
    """
    Create a heatmap of accuracy across configurations.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
    """
    # Create pivot table for heatmap
    # Filter for specific sequence length
    filtered_df = metrics_df[metrics_df['Seq Length'] == 50]
    
    if len(filtered_df) > 0:
        # Create a combination column
        filtered_df = filtered_df.copy()
        filtered_df['Config'] = (filtered_df['Model'] + '_' + 
                                filtered_df['Activation'] + '_' + 
                                filtered_df['Grad Clipping'])
        
        # Pivot table
        pivot_data = filtered_df.pivot_table(
            values='Accuracy',
            index='Optimizer',
            columns='Config',
            aggfunc='first'
        )
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot_data.astype(float), annot=True, fmt='.2f', 
                   cmap='YlOrRd', cbar_kws={'label': 'Accuracy (%)'})
        plt.title('Accuracy Heatmap (Sequence Length = 50)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel('Optimizer', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('results/plots/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: results/plots/accuracy_heatmap.png")


def generate_summary_report(metrics_df, detailed_results):
    """
    Generate a summary report.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
        detailed_results (list): Detailed results
    """
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    # Best overall configuration
    best_idx = metrics_df['Accuracy'].astype(float).idxmax()
    best_config = metrics_df.iloc[best_idx]
    
    print("\nBest Configuration:")
    print(f"  Model: {best_config['Model']}")
    print(f"  Activation: {best_config['Activation']}")
    print(f"  Optimizer: {best_config['Optimizer']}")
    print(f"  Sequence Length: {best_config['Seq Length']}")
    print(f"  Gradient Clipping: {best_config['Grad Clipping']}")
    print(f"  Accuracy: {best_config['Accuracy']}%")
    print(f"  F1 Score: {best_config['F1']}")
    print(f"  Avg Epoch Time: {best_config['Epoch Time (s)']}s")
    
    # Architecture comparison
    print("\n" + "-"*80)
    print("Architecture Performance (baseline config):")
    baseline = metrics_df[
        (metrics_df['Activation'] == 'tanh') & 
        (metrics_df['Optimizer'] == 'adam') & 
        (metrics_df['Seq Length'] == 50) &
        (metrics_df['Grad Clipping'] == 'No')
    ]
    for _, row in baseline.iterrows():
        print(f"  {row['Model']}: {row['Accuracy']}% accuracy, {row['F1']} F1")
    
    # Optimizer comparison
    print("\n" + "-"*80)
    print("Optimizer Performance (LSTM, tanh, seq len 50):")
    opt_comparison = metrics_df[
        (metrics_df['Model'] == 'LSTM') & 
        (metrics_df['Activation'] == 'tanh') & 
        (metrics_df['Seq Length'] == 50) &
        (metrics_df['Grad Clipping'] == 'No')
    ]
    for _, row in opt_comparison.iterrows():
        print(f"  {row['Optimizer'].upper()}: {row['Accuracy']}% accuracy, "
              f"{row['Epoch Time (s)']}s per epoch")
    
    # Sequence length impact
    print("\n" + "-"*80)
    print("Sequence Length Impact (LSTM, adam, tanh):")
    seq_comparison = metrics_df[
        (metrics_df['Model'] == 'LSTM') & 
        (metrics_df['Activation'] == 'adam') & 
        (metrics_df['Optimizer'] == 'adam') &
        (metrics_df['Grad Clipping'] == 'No')
    ]
    for _, row in seq_comparison.iterrows():
        print(f"  Length {row['Seq Length']}: {row['Accuracy']}% accuracy")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    print("Loading results...")
    
    # Create plots directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Load results
    metrics_df, detailed_results = load_results()
    
    print(f"Loaded {len(metrics_df)} experimental results\n")
    
    # Generate plots
    print("Generating plots...")
    plot_accuracy_vs_seq_length(metrics_df)
    plot_f1_vs_seq_length(metrics_df)
    plot_model_comparison(metrics_df)
    plot_optimizer_comparison(metrics_df)
    plot_gradient_clipping_effect(metrics_df)
    plot_heatmap(metrics_df)
    
    if detailed_results:
        plot_training_curves(detailed_results)
    
    # Generate summary report
    generate_summary_report(metrics_df, detailed_results)
    
    print("\n" + "="*80)
    print("Evaluation complete! All plots saved to results/plots/")
    print("="*80)


if __name__ == "__main__":
    main()