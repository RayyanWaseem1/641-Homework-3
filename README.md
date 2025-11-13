# RNN Sentiment Classification - Reproducibility Guide

## Overview
Comparative analysis of RNN architectures (RNN, LSTM, BiLSTM) for sentiment classification on the IMDb dataset. Achieves 79.89% accuracy with optimized LSTM configuration.

## Requirements
- Python 3.8+
- 8GB RAM minimum
- ~500MB disk space
- CPU only (no GPU required)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Edit `src/preprocess.py` line 168 to point to your IMDb CSV file:
```python
csv_path = os.path.expanduser("~/Desktop/IMDB Dataset.csv")
```

If your CSV has different column names, edit lines 180-181:
```python
text_column = 'review'      # Your text column name
label_column = 'sentiment'  # Your label column name
```

Run preprocessing:
```bash
python src/preprocess.py
```
Expected time: 5-10 minutes. Creates `data/preprocessed_complete.pkl`.

### 3. Run Experiments
```bash
# Quick test (1 epoch, ~5 minutes)
python src/train.py --model lstm --optimizer adam --seq-length 50 --epochs 1

# All experiments (5 epochs each, 4-6 hours)
python src/train.py --run-all
```

Expected output: `results/metrics.csv` with 15 experimental configurations.

### 4. Generate Analysis
```bash
python src/evaluate.py
```
Creates: `results/plots/` (7 visualization files)

## Reproducibility

### Fixed Seeds
All random seeds set to 42:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Hardware Used
- **CPU:** Apple M1/M2 or Intel equivalent
- **RAM:** 16GB
- **OS:** macOS 13+

Results may vary slightly on different hardware but should be within ±1% accuracy.

### Expected Results
| Configuration | Accuracy | F1 Score | Epoch Time |
|--------------|----------|----------|------------|
| Best (LSTM-sigmoid-rmsprop-100-clip) | 79.89% | 0.7985 | 19.52s |
| Baseline (LSTM-tanh-adam-50) | 76.13% | 0.7609 | 10.71s |
| Worst (RNN-relu-sgd-50-clip) | 50.84% | 0.4895 | 4.09s |

### Model Configurations
All models use:
- Embedding dimension: 100
- Hidden layers: 2
- Hidden size: 64
- Dropout: 0.4
- Batch size: 32
- Epochs: 5
- Learning rate: 0.001

### Experimental Variations Tested
- **Architectures:** RNN, LSTM, Bidirectional LSTM
- **Activations:** sigmoid, tanh, ReLU
- **Optimizers:** Adam, SGD (momentum=0.9), RMSProp
- **Sequence Lengths:** 25, 50, 100
- **Gradient Clipping:** None vs. max_norm=1.0

## File Structure
```
├── src/
│   ├── preprocess.py      # Data preprocessing
│   ├── models.py          # Model architectures
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation & plots
│   └── utils.py           # Helper functions
├── data/                  # Generated data files
├── results/
│   ├── metrics.csv        # All experimental results
│   └── plots/             # 7 visualization files
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Dataset Format
CSV file with two columns:
- Text column: Movie reviews (any length)
- Label column: `positive`/`negative` or `1`/`0`

Example:
```csv
review,sentiment
"Great movie!",positive
"Terrible film",negative
```

Standard IMDb dataset: 50,000 reviews (25k train, 25k test)

## Troubleshooting

**ImportError:** Run `pip install -r requirements.txt`

**Out of memory:** Reduce batch size in `train.py` line 204: `batch_size=16`

**Slow training:** Use shorter sequences: `--seq-length 25` or fewer epochs: `--epochs 3`

**CSV not found:** Check path in `preprocess.py` line 168

## License
MIT License - Free to use for academic and research purposes.
