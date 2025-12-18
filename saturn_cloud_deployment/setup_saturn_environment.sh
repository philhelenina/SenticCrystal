#!/bin/bash

# Saturn Cloud A100 Environment Setup for SenticCrystal
# Run this script to set up the complete environment

set -e

echo "🚀 Setting up SenticCrystal environment on Saturn Cloud A100..."
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create conda environment with mamba (faster)
echo "📦 Creating conda environment with mamba..."
mamba env create -f environment.yml

# Activate environment
echo "🔧 Activating environment..."
conda activate senticcrystal

# Verify CUDA availability
echo "🔍 Verifying CUDA setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"

# Download required NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Verify sentence-transformers
echo "🔍 Verifying sentence-transformers..."
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('nli-distilroberta-base-v2'); print('✅ Sentence-RoBERTa model loaded successfully')"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/iemocap_4way_data
mkdir -p scripts/embeddings
mkdir -p results/baseline_classifiers
mkdir -p results/turn_experiments
mkdir -p src/models
mkdir -p src/data_preprocessing

echo "✅ Environment setup complete!"
echo "💡 To activate: conda activate senticcrystal"
echo "🎯 Ready for SenticCrystal experiments!"