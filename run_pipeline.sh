#!/bin/bash

# Quick Start Script for RNN Sentiment Classification Project
# This script runs the entire pipeline from preprocessing to evaluation

echo "=========================================="
echo "RNN Sentiment Classification Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "Step 1: Data Preprocessing"
echo "=========================================="
echo ""

# Run preprocessing
python src/preprocess.py

echo ""
echo "=========================================="
echo "Step 2: Training Models"
echo "=========================================="
echo ""

# Ask user if they want to run all experiments
read -p "Run all experiments? This may take several hours. (y/n): " run_all

if [ "$run_all" = "y" ] || [ "$run_all" = "Y" ]; then
    python src/train.py --run-all
else
    echo "Running single experiment (LSTM, Adam, seq length 50)..."
    python src/train.py --model lstm --optimizer adam --seq-length 50 --epochs 5
fi

echo ""
echo "=========================================="
echo "Step 3: Evaluation and Visualization"
echo "=========================================="
echo ""

# Run evaluation
python src/evaluate.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/metrics.csv"
echo "  - results/plots/"
echo ""
echo "Next steps:"
echo "  1. Review the results in results/metrics.csv"
echo "  2. Check the visualizations in results/plots/"
echo "  3. Write your report based on the findings"
echo ""