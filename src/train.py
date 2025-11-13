"""
Training script for RNN sentiment classification models.
Handles training loop, evaluation, and experiment management.
"""

import os
import time
import argparse
import pickle
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models import get_model
from preprocess import create_dataloaders
from utils import (set_seed, get_device, create_directories, 
                   count_parameters, get_experiment_name, AverageMeter)


def get_optimizer(optimizer_name, model_parameters, learning_rate=0.001):
    """
    Get optimizer based on name.
    
    Args:
        optimizer_name (str): Name of optimizer ('adam', 'sgd', 'rmsprop')
        model_parameters: Model parameters
        learning_rate (float): Learning rate
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping value (None for no clipping)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Calculate metrics
        losses.update(loss.item(), sequences.size(0))
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    accuracy = 100.0 * correct / total
    return losses.avg, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        for batch in progress_bar:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            losses.update(loss.item(), sequences.size(0))
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    accuracy = 100.0 * correct / total
    return losses.avg, accuracy, all_predictions, all_labels


def train_model(model_type, activation, optimizer_name, seq_length, 
                grad_clip, vocab_size, num_epochs=5, batch_size=32,
                learning_rate=0.001):
    """
    Train a single model configuration.
    
    Args:
        model_type (str): Type of model ('rnn', 'lstm', 'bilstm')
        activation (str): Activation function
        optimizer_name (str): Optimizer name
        seq_length (int): Sequence length
        grad_clip (bool): Whether to use gradient clipping
        vocab_size (int): Vocabulary size
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        
    Returns:
        dict: Results dictionary
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open('data/preprocessed_complete.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Get data for specific sequence length
    seq_data = data['processed_data'][seq_length]
    train_sequences = seq_data['train_sequences']
    train_labels = seq_data['train_labels']
    test_sequences = seq_data['test_sequences']
    test_labels = seq_data['test_labels']
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloaders(train_sequences, train_labels, 
                                     batch_size=batch_size, shuffle=True)
    test_loader = create_dataloaders(test_sequences, test_labels, 
                                    batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    model = get_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.4,
        activation=activation
    )
    model = model.to(device)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    
    # Gradient clipping value
    clip_value = 1.0 if grad_clip else None
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_value
        )
        epoch_time = time.time() - epoch_start_time
        
        # Evaluate on test set
        test_loss, test_acc, test_preds, test_true = evaluate(
            model, test_loader, criterion, device
        )
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        epoch_times.append(epoch_time)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(test_true, test_preds, average='macro')
    
    # Prepare results
    results = {
        'model_type': model_type,
        'activation': activation,
        'optimizer': optimizer_name,
        'seq_length': seq_length,
        'grad_clip': 'Yes' if grad_clip else 'No',
        'test_accuracy': test_accuracies[-1],
        'test_f1': f1,
        'avg_epoch_time': np.mean(epoch_times),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Avg Epoch Time: {np.mean(epoch_times):.2f}s")
    
    return results


def run_all_experiments():
    """
    Run all experimental configurations.
    """
    # Load vocabulary size
    with open('data/preprocessed_complete.pkl', 'rb') as f:
        data = pickle.load(f)
    vocab_size = data['vocab_size']
    
    # Define experimental configurations
    # For a complete comparison, we test key combinations
    configurations = [
        # Baseline configurations for each architecture
        ('rnn', 'tanh', 'adam', 50, False),
        ('lstm', 'tanh', 'adam', 50, False),
        ('bilstm', 'tanh', 'adam', 50, False),
        
        # Test different activations (LSTM)
        ('lstm', 'sigmoid', 'adam', 50, False),
        ('lstm', 'relu', 'adam', 50, False),
        
        # Test different optimizers (LSTM)
        ('lstm', 'tanh', 'sgd', 50, False),
        ('lstm', 'tanh', 'rmsprop', 50, False),
        
        # Test different sequence lengths (LSTM)
        ('lstm', 'tanh', 'adam', 25, False),
        ('lstm', 'tanh', 'adam', 100, False),
        
        # Test gradient clipping (all architectures)
        ('rnn', 'tanh', 'adam', 50, True),
        ('lstm', 'tanh', 'adam', 50, True),
        ('bilstm', 'tanh', 'adam', 50, True),
        
        # Additional combinations for thorough analysis
        ('bilstm', 'relu', 'adam', 50, False),
        ('rnn', 'relu', 'sgd', 50, True),
        ('lstm', 'sigmoid', 'rmsprop', 100, True),
    ]
    
    # Results storage
    all_results = []
    
    # Run each configuration
    for i, (model_type, activation, optimizer, seq_length, grad_clip) in enumerate(configurations):
        print("\n" + "="*80)
        print(f"Experiment {i + 1}/{len(configurations)}")
        print(f"Model: {model_type.upper()} | Activation: {activation} | "
              f"Optimizer: {optimizer} | Seq Length: {seq_length} | "
              f"Grad Clip: {'Yes' if grad_clip else 'No'}")
        print("="*80)
        
        try:
            results = train_model(
                model_type=model_type,
                activation=activation,
                optimizer_name=optimizer,
                seq_length=seq_length,
                grad_clip=grad_clip,
                vocab_size=vocab_size,
                num_epochs=5
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError in experiment: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    print("\n" + "="*80)
    print("Saving results to CSV...")
    
    csv_file = 'results/metrics.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['Model', 'Activation', 'Optimizer', 'Seq Length', 
                     'Grad Clipping', 'Accuracy', 'F1', 'Epoch Time (s)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                'Model': result['model_type'].upper(),
                'Activation': result['activation'],
                'Optimizer': result['optimizer'],
                'Seq Length': result['seq_length'],
                'Grad Clipping': result['grad_clip'],
                'Accuracy': f"{result['test_accuracy']:.2f}",
                'F1': f"{result['test_f1']:.4f}",
                'Epoch Time (s)': f"{result['avg_epoch_time']:.2f}"
            })
    
    print(f"Results saved to {csv_file}")
    
    # Save detailed results
    with open('results/detailed_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("Detailed results saved to results/detailed_results.pkl")
    print("="*80)
    print("\nAll experiments completed!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train RNN models for sentiment classification')
    
    parser.add_argument('--run-all', action='store_true',
                       help='Run all experimental configurations')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['rnn', 'lstm', 'bilstm'],
                       help='Model architecture')
    parser.add_argument('--activation', type=str, default='tanh',
                       choices=['sigmoid', 'relu', 'tanh'],
                       help='Activation function')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--seq-length', type=int, default=50,
                       choices=[25, 50, 100],
                       help='Sequence length')
    parser.add_argument('--grad-clip', action='store_true',
                       help='Use gradient clipping')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Check if preprocessed data exists
    if not os.path.exists('data/preprocessed_complete.pkl'):
        print("Preprocessed data not found!")
        print("Please run: python src/preprocess.py")
        return
    
    if args.run_all:
        # Run all experiments
        run_all_experiments()
    else:
        # Run single experiment
        with open('data/preprocessed_complete.pkl', 'rb') as f:
            data = pickle.load(f)
        vocab_size = data['vocab_size']
        
        results = train_model(
            model_type=args.model,
            activation=args.activation,
            optimizer_name=args.optimizer,
            seq_length=args.seq_length,
            grad_clip=args.grad_clip,
            vocab_size=vocab_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )


if __name__ == "__main__":
    main()
