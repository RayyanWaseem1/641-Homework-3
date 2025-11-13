"""
Utility functions for the RNN sentiment classification project.
"""

import torch
import random
import numpy as np
import os
import json
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the available device (CUDA or CPU).
    
    Returns:
        torch.device: Available device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'results',
        'results/plots',
        'results/models'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully!")


def save_config(config, filename='config.json'):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        filename (str): Output filename
    """
    with open(f'results/{filename}', 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to results/{filename}")


def load_config(filename='config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        filename (str): Input filename
        
    Returns:
        dict: Configuration dictionary
    """
    with open(f'results/{filename}', 'r') as f:
        config = json.load(f)
    return config


def format_time(seconds):
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(model)
    print("-"*60)
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("="*60 + "\n")


def get_experiment_name(model_type, activation, optimizer, seq_length, grad_clip):
    """
    Generate a unique experiment name based on configuration.
    
    Args:
        model_type (str): Model architecture (rnn, lstm, bilstm)
        activation (str): Activation function
        optimizer (str): Optimizer name
        seq_length (int): Sequence length
        grad_clip (bool): Whether gradient clipping is used
        
    Returns:
        str: Experiment name
    """
    clip_str = "clip" if grad_clip else "noclip"
    return f"{model_type}_{activation}_{optimizer}_len{seq_length}_{clip_str}"


def log_experiment(experiment_name, metrics, log_file='results/experiment_log.txt'):
    """
    Log experiment results to a file.
    
    Args:
        experiment_name (str): Name of the experiment
        metrics (dict): Dictionary of metrics
        log_file (str): Path to log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'-'*80}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"{'='*80}\n")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_progress(current, total, prefix='', suffix='', bar_length=50):
    """
    Print a progress bar to console.
    
    Args:
        current (int): Current iteration
        total (int): Total iterations
        prefix (str): Prefix string
        suffix (str): Suffix string
        bar_length (int): Length of progress bar
    """
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = 100 * (current / total)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if current == total:
        print()
        