"""
Data preprocessing for IMDb sentiment classification.
Handles downloading, cleaning, tokenization, and sequence preparation.
"""

import os
import re
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# For downloading IMDb dataset
try:
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer
except ImportError:
    print("torchtext not available, will use alternative download method")


class IMDbDataset(Dataset):
    """Custom Dataset for IMDb reviews"""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


class IMDbPreprocessor:
    """
    Preprocessor for IMDb Movie Review Dataset.
    """
    
    def __init__(self, vocab_size=10000, seq_lengths=[25, 50, 100]):
        """
        Initialize the preprocessor.
        
        Args:
            vocab_size (int): Maximum vocabulary size
            seq_lengths (list): List of sequence lengths to prepare
        """
        self.vocab_size = vocab_size
        self.seq_lengths = seq_lengths
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_freq = Counter()
        
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation and special characters, keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return text.split()
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts.
        
        Args:
            texts (list): List of text strings
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in tqdm(texts, desc="Counting words"):
            tokens = self.tokenize(self.clean_text(text))
            self.vocab_freq.update(tokens)
        
        # Keep top vocab_size - 2 words (accounting for PAD and UNK)
        most_common = self.vocab_freq.most_common(self.vocab_size - 2)
        
        # Build word2idx and idx2word mappings
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {most_common[:10]}")
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of token IDs.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of token IDs
        """
        tokens = self.tokenize(self.clean_text(text))
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        return sequence
    
    def pad_sequence(self, sequence, max_length):
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            sequence (list): Input sequence
            max_length (int): Target length
            
        Returns:
            list: Padded/truncated sequence
        """
        if len(sequence) > max_length:
            # Truncate
            return sequence[:max_length]
        else:
            # Pad
            padding = [self.word2idx['<PAD>']] * (max_length - len(sequence))
            return sequence + padding
    
    def save_preprocessed_data(self, data, filename):
        """Save preprocessed data to file."""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {filename}")
    
    def load_preprocessed_data(self, filename):
        """Load preprocessed data from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


def download_imdb_dataset():
    """
    Load IMDb dataset from CSV file.
    
    Returns:
        tuple: (train_texts, train_labels, test_texts, test_labels)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # ==================== EDIT THIS PATH ====================
    csv_path = os.path.expanduser("~/Desktop/rnn-sentiment-analysis/IMDB Dataset.csv")
    # ========================================================
    
    print(f"Loading dataset from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # ==================== EDIT THESE COLUMN NAMES IF NEEDED ====================
    text_column = 'review'      # Column containing the text/review
    label_column = 'sentiment'  # Column containing the label/sentiment
    # ===========================================================================
    
    # Extract data
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Convert labels to binary (0 and 1)
    binary_labels = []
    for label in labels:
        label_str = str(label).lower()
        if label_str in ['positive', 'pos', '1', '1.0']:
            binary_labels.append(1)
        elif label_str in ['negative', 'neg', '0', '0.0']:
            binary_labels.append(0)
        else:
            # Default to treating as positive if > 0, negative otherwise
            try:
                binary_labels.append(1 if float(label) > 0 else 0)
            except:
                binary_labels.append(0)
    
    print(f"Positive samples: {sum(binary_labels)}")
    print(f"Negative samples: {len(binary_labels) - sum(binary_labels)}")
    
    # Split into train/test (50/50 split)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, binary_labels, test_size=0.5, random_state=42, stratify=binary_labels
    )
    
    print(f"\nDataset split complete!")
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return train_texts, train_labels, test_texts, test_labels


def create_dataloaders(sequences, labels, batch_size=32, shuffle=True):
    """
    Create DataLoader from sequences and labels.
    
    Args:
        sequences (list): List of sequences
        labels (list): List of labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = IMDbDataset(sequences, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True
    )
    return dataloader


def preprocess_and_save():
    """
    Main preprocessing pipeline.
    Downloads, preprocesses, and saves the IMDb dataset.
    """
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Check if already preprocessed
    if os.path.exists('data/preprocessed_complete.pkl'):
        print("Preprocessed data already exists!")
        response = input("Do you want to reprocess? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Download dataset
    train_texts, train_labels, test_texts, test_labels = download_imdb_dataset()
    
    # Initialize preprocessor
    preprocessor = IMDbPreprocessor(vocab_size=10000, seq_lengths=[25, 50, 100])
    
    # Build vocabulary from training data
    preprocessor.build_vocabulary(train_texts)
    
    # Calculate statistics
    train_lengths = [len(preprocessor.tokenize(preprocessor.clean_text(text))) 
                    for text in train_texts[:1000]]  # Sample for speed
    print(f"\nDataset statistics (sample):")
    print(f"Average review length: {np.mean(train_lengths):.2f} words")
    print(f"Median review length: {np.median(train_lengths):.2f} words")
    print(f"Max review length: {np.max(train_lengths)} words")
    print(f"Min review length: {np.min(train_lengths)} words")
    
    # Process and save data for each sequence length
    processed_data = {}
    
    for seq_length in preprocessor.seq_lengths:
        print(f"\nProcessing sequences for length {seq_length}...")
        
        # Convert texts to sequences
        print("Converting training texts...")
        train_sequences = [preprocessor.text_to_sequence(text) for text in tqdm(train_texts)]
        print("Converting test texts...")
        test_sequences = [preprocessor.text_to_sequence(text) for text in tqdm(test_texts)]
        
        # Pad sequences
        print(f"Padding sequences to length {seq_length}...")
        train_sequences_padded = [preprocessor.pad_sequence(seq, seq_length) 
                                 for seq in tqdm(train_sequences)]
        test_sequences_padded = [preprocessor.pad_sequence(seq, seq_length) 
                                for seq in tqdm(test_sequences)]
        
        # Store in dictionary
        processed_data[seq_length] = {
            'train_sequences': train_sequences_padded,
            'train_labels': train_labels,
            'test_sequences': test_sequences_padded,
            'test_labels': test_labels
        }
    
    # Save everything
    complete_data = {
        'word2idx': preprocessor.word2idx,
        'idx2word': preprocessor.idx2word,
        'vocab_size': len(preprocessor.word2idx),
        'processed_data': processed_data
    }
    
    preprocessor.save_preprocessed_data(complete_data, 'data/preprocessed_complete.pkl')
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Vocabulary size: {len(preprocessor.word2idx)}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Sequence lengths prepared: {preprocessor.seq_lengths}")
    print("="*60)


if __name__ == "__main__":
    preprocess_and_save()