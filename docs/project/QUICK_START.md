# 🚀 SenticCrystal Quick Start Guide

Get up and running with SenticCrystal emotion recognition in minutes!

## 🎯 **Prerequisites**

- Python 3.8+ (3.10 recommended)
- CUDA-compatible GPU (optional, but recommended)
- 16GB+ RAM (32GB+ for large experiments)
- Git

## ⚡ **5-Minute Setup**

### **Step 1: Clone & Navigate**
```bash
git clone <repository-url>
cd SenticCrystal
```

### **Step 2: Environment Setup**

#### **Option A: Local Development (MacBook/CPU)**
```bash
# Create virtual environment
python -m venv senticcrystal
source senticcrystal/bin/activate  # Mac/Linux
# OR
senticcrystal\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers datasets
pip install numpy pandas scikit-learn matplotlib seaborn
pip install nltk xmltodict
```

#### **Option B: GPU Development (CUDA)**
```bash
# Create virtual environment
python -m venv senticcrystal
source senticcrystal/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install sentence-transformers transformers datasets
pip install numpy pandas scikit-learn matplotlib seaborn
pip install nltk xmltodict
```

#### **Option C: Saturn Cloud A100 (Production)**
```bash
# Upload environment file to Saturn Cloud
# Then create conda environment
conda env create -f docs/setup/environment_saturn_cloud.yml
conda activate senticcrystal-saturn
```

### **Step 3: Download NLTK Data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 🔮 **Basic Usage Examples**

### **Example 1: Simple Emotion Recognition**
```python
from src.data_preprocessing.config146_generator import Config146EmbeddingGenerator
import pandas as pd

# Sample conversation data
texts = [
    "Hello, how are you doing today?",
    "I'm feeling really frustrated with this situation.",
    "That's wonderful news! I'm so happy for you.",
    "I'm sorry to hear that. It must be difficult."
]
ids = ["utt_001", "utt_002", "utt_003", "utt_004"]

# Initialize generator (auto-detects best device)
generator = Config146EmbeddingGenerator()

# Generate embeddings with context
embeddings = generator.generate_embeddings(
    texts=texts,
    ids=ids,
    context_turns=4  # Use 4-turn context window
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings[0][0].shape}")  # (context_turns, 768)
```

### **Example 2: Multi-K Context Analysis**
```python
# Generate multiple K values efficiently (4-5x faster than separate calls)
multi_k_embeddings = generator.generate_multiple_k_embeddings(
    texts=texts,
    ids=ids,
    k_values=[0, 2, 4, 6]  # Different context window sizes
)

# Access different K values
k0_embeddings = multi_k_embeddings[0]  # Current utterance only
k2_embeddings = multi_k_embeddings[2]  # Current + 1 previous
k4_embeddings = multi_k_embeddings[4]  # Current + 3 previous  
k6_embeddings = multi_k_embeddings[6]  # Current + 5 previous

print("Multi-K generation complete!")
for k, embs in multi_k_embeddings.items():
    print(f"K={k}: {len(embs)} embeddings, shape: {embs[0][0].shape}")
```

### **Example 3: Bayesian Uncertainty Quantification**
```python
from src.data_preprocessing.bayesian_config146_generator import BayesianConfig146EmbeddingGenerator

# Initialize Bayesian generator
bayesian_gen = BayesianConfig146EmbeddingGenerator(
    dropout=0.3,        # Monte Carlo Dropout rate
    mc_samples=10       # Number of MC samples for uncertainty
)

# Generate with uncertainty information
embeddings, uncertainty_info = bayesian_gen.generate_embeddings(
    texts=texts,
    ids=ids,
    return_uncertainty=True
)

# Examine uncertainty information
for i, info in enumerate(uncertainty_info):
    print(f"Utterance {i}: confidence={info['confidence_score']:.3f}, "
          f"level={info['confidence_level']}, "
          f"needs_review={info['needs_human_review']}")
```

### **Example 4: Confidence-based Quality Control**
```python
# Filter by confidence threshold
high_confidence, low_confidence_ids, all_uncertainty = bayesian_gen.generate_with_confidence_filtering(
    texts=texts,
    ids=ids,
    confidence_threshold=0.8  # Only keep 80%+ confidence
)

print(f"High confidence samples: {len(high_confidence)}")
print(f"Low confidence samples requiring review: {len(low_confidence_ids)}")
print(f"Retention rate: {len(high_confidence)/len(texts)*100:.1f}%")
```

## 📊 **Running Full Experiments**

### **Generate Config146 Embeddings**
```bash
# Generate embeddings for all K values (K=0,2,4,6)
python scripts/embeddings.py

# Output will be saved to embeddings/config146_proper/
# ├── 0turn/  - K=0 embeddings
# ├── 2turn/  - K=2 embeddings  
# ├── 4turn/  - K=4 embeddings
# └── 6turn/  - K=6 embeddings
```

### **Run Complete Turn Analysis**
```bash
# Run comprehensive experiments with all strategies
python run_comprehensive_experiments.py

# This will:
# 1. Load pre-generated embeddings
# 2. Test baseline strategies (K=0,2,4,6)
# 3. Test cumulative strategies (pure, conservative, quantile)
# 4. Generate performance reports and visualizations
# 5. Save results to results/ directory
```

### **Expected Output**
```
🎯 Experiment Results Summary:
┌─────────────────────┬──────────┬──────────┬─────────────┐
│ Strategy            │ Accuracy │ Macro-F1 │ Weighted-F1 │
├─────────────────────┼──────────┼──────────┼─────────────┤
│ Baseline K=0        │  66.2%   │  66.1%   │    66.3%    │
│ Baseline K=2        │  67.8%   │  67.6%   │    67.9%    │  
│ Baseline K=4        │  70.1%   │  69.9%   │    70.2%    │
│ Config146 K=6       │  72.0%   │  71.9%   │    72.4%    │ ← Target!
│ Cumulative Quantile │  74.2%   │  74.1%   │    74.5%    │ ← Best!
└─────────────────────┴──────────┴──────────┴─────────────┘
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. CUDA Out of Memory**
```python
# Reduce batch size in your code
batch_size = 16  # Instead of 32 or 64

# Or use CPU fallback
generator = Config146EmbeddingGenerator(device='cpu')
```

#### **2. WordNet-Affect Not Found**
```python
# Check WordNet-Affect path
import os
wn_path = "scripts/wn-affect-1.0"
print(f"WordNet path exists: {os.path.exists(wn_path)}")

# If missing, check backup directory
backup_path = "backup/wordnet_data"
if os.path.exists(backup_path):
    print("WordNet data found in backup, may need to restore")
```

#### **3. Module Import Errors**
```python
# Add project root to Python path
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Then import
from src.data_preprocessing.config146_generator import Config146EmbeddingGenerator
```

#### **4. Slow Performance**
```python
# Check device detection
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")  # Mac M1/M2/M4

# Force device selection
generator = Config146EmbeddingGenerator(device='cuda')  # or 'mps' for Mac
```

### **Performance Optimization Tips**

#### **For MacBook M1/M2/M4**
```python
# Use MPS acceleration
generator = Config146EmbeddingGenerator(device='mps')

# Optimize batch size for Apple Silicon
config = {'batch_size': 16}  # Sweet spot for M-series chips
```

#### **For NVIDIA GPUs**
```python
# Use CUDA with larger batches
generator = Config146EmbeddingGenerator(device='cuda')

# For RTX 4090 / A100
config = {'batch_size': 64}  # Can handle larger batches
```

## 📖 **Next Steps**

### **Learn More**
- 📊 [Experimental Results](docs/experiments/EXPERIMENTAL_RESULTS_SUMMARY.md) - Detailed performance analysis
- 🔄 [Turn Analysis](docs/experiments/COMPREHENSIVE_TURN_ANALYSIS_PLAN.md) - Context modeling strategies  
- 🧠 [Bayesian Methods](docs/analysis/REFACTORING_COMPLETE.md) - Uncertainty quantification
- ☁️ [Saturn Cloud Setup](docs/setup/saturn_cloud_setup.md) - High-performance training

### **Customize & Extend**
```python
# Custom configuration
custom_config = {
    'apply_word_pe': True,           # Enable word positional encoding
    'pooling_method': 'max',         # Try different pooling
    'combination_method': 'concatenate',  # Instead of sum
    'alpha': 0.7,                    # Adjust WordNet-Affect weight
    'context_turns': 8               # Larger context window
}

generator = Config146EmbeddingGenerator(config=custom_config)
```

### **Production Deployment**
For production use with real-time requirements:
1. Pre-generate embeddings for known datasets
2. Use confidence filtering to ensure quality
3. Implement caching for repeated inputs
4. Monitor uncertainty scores for data drift

---

🎉 **You're ready to start using SenticCrystal!** 

For issues or questions, check the [troubleshooting](#-troubleshooting) section or open an issue on GitHub.