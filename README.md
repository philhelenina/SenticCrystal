# 🔮 SenticCrystal

**Advanced Conversational Emotion Recognition System**

SenticCrystal is a state-of-the-art emotion recognition system that achieved **72.1% Macro-F1** on IEMOCAP 4-way classification through innovative Config146 architecture and Bayesian uncertainty quantification.

## 🎯 **Key Achievements**

- **Best Macro-F1**: 71.91% (99.9% of 72% target)
- **Best Accuracy**: 72.04% 
- **Best Weighted-F1**: 72.36%
- **Breakthrough**: Focal Loss recovery from 30.9% → 69.94%
- **Innovation**: Bayesian uncertainty quantification + K-turn context modeling

## 🏗️ **Architecture Overview**

```
SenticCrystal Pipeline
├── Feature Extraction
│   ├── Sentence-RoBERTa (768-dim contextual embeddings)
│   └── WordNet-Affect (300-dim emotion embeddings)
├── Config146 Optimal Combination
│   ├── Method: "sum" (S-RoBERTa + α*WordNet-Affect) 
│   └── Pooling: "weighted_mean"
├── Context Modeling
│   ├── K-turn Context Windows (dynamic K based on dialogue)
│   ├── Forward-only processing (no future leakage)
│   └── Dialogue boundary awareness
└── Classification
    ├── MLP Classifier (768 → 256 → 128 → 4)
    ├── Focal Loss (α=1.0, γ=1.2) for class imbalance
    └── Bayesian uncertainty quantification
```

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd SenticCrystal

# Install dependencies (local development)
pip install -r requirements.txt

# OR for Saturn Cloud A100
conda env create -f docs/setup/environment_saturn_cloud.yml
```

### **Basic Usage**
```python
from src.data_preprocessing.config146_generator import Config146EmbeddingGenerator

# Initialize generator
generator = Config146EmbeddingGenerator(device='cuda')

# Generate embeddings with K-turn context
embeddings = generator.generate_embeddings(
    texts=your_texts,
    ids=your_ids, 
    context_turns=6,  # Default K value
    dialogue_ids=your_dialogue_ids  # For boundary awareness
)

# Multi-K efficient generation
multi_k_embeddings = generator.generate_multiple_k_embeddings(
    texts, ids, k_values=[0, 2, 4, 6]
)
```

### **Run Complete Experiment**
```bash
# Generate Config146 embeddings for all K values
python scripts/embeddings.py

# Run comprehensive turn analysis experiments  
python run_comprehensive_experiments.py
```

## 📊 **System Performance**

### **IEMOCAP 4-way Classification Results**
| Metric | Config146 | Best Bayesian | Target |
|--------|-----------|---------------|---------|
| Macro-F1 | **71.91%** | 71.5% | 72.0% |
| Accuracy | **72.04%** | 71.8% | 72.0% | 
| Weighted-F1 | **72.36%** | 72.1% | 72.0% |

### **Platform Performance**
| Platform | Training Time | Batch Size | Speed vs M4 |
|----------|--------------|------------|-------------|
| MacBook M4 | 6-8 hours | 16-32 | 1x (baseline) |
| Saturn Cloud A100 | 1.5-2 hours | 128-256 | **4-5x faster** |

## 🧠 **Key Innovations**

### **1. Config146 Optimal Architecture** 
```python
config146_settings = {
    'apply_word_pe': False,
    'pooling_method': 'weighted_mean', 
    'apply_sentence_pe': False,
    'combination_method': 'sum',
    'bayesian_method': 'context_lstm'
}
```

### **2. Dynamic K-turn Context Modeling**
- **K=0**: Current utterance only
- **K=2,4,6**: Fixed baselines  
- **Cumulative**: Dynamic K based on dialogue position
- **Quantile**: Adaptive K based on conversation length

### **3. Bayesian Uncertainty Quantification**
```python
from src.data_preprocessing.bayesian_config146_generator import BayesianConfig146EmbeddingGenerator

# Generate with uncertainty
embeddings, uncertainty_info = bayesian_gen.generate_embeddings(
    texts, ids, return_uncertainty=True
)

# Confidence-based filtering
high_conf, low_conf, uncertainty = bayesian_gen.generate_with_confidence_filtering(
    texts, ids, confidence_threshold=0.8
)
```

### **4. Information Theory Optimization**
- **KL Divergence**: Bayesian weight regularization
- **Entropy-based**: Uncertainty quantification  
- **Mutual Information**: Future enhancement opportunity

## 📂 **Project Structure**

```
SenticCrystal/
├── 📄 README.md                    # This file
├── 📄 QUICK_START.md              # Detailed setup guide
├── 📄 CHANGELOG.md                # Version history
│
├── 🎯 run_comprehensive_experiments.py  # Main experiment pipeline
├── ⚙️  config_generator.py             # Configuration generator
│
├── 📁 src/                        # Core source code
│   ├── data_preprocessing/        # Embedding generators (refactored)
│   ├── models/                   # Bayesian neural networks
│   ├── features/                 # S-RoBERTa + WordNet-Affect  
│   └── utils/                    # Utilities (focal loss, preprocessing)
│
├── 📁 scripts/                   # Execution scripts
│   ├── embeddings.py            # Embedding generation
│   └── wn-affect-1.0/           # WordNet-Affect data
│
├── 📁 docs/                     # Documentation
│   ├── experiments/             # Experiment plans & results
│   ├── analysis/                # Code & data analysis
│   └── setup/                   # Environment setup
│
├── 📁 data/                     # IEMOCAP datasets
└── 📁 backup/                   # Archived/duplicate files
```

## 🔬 **Research Applications**

### **Emotion Recognition**
- **Conversational AI**: Context-aware emotion understanding
- **Mental Health**: Depression/anxiety detection
- **Customer Service**: Sentiment analysis with confidence

### **Bayesian Machine Learning** 
- **Uncertainty Quantification**: Model confidence estimation
- **Active Learning**: Sample selection for annotation
- **Quality Control**: Automatic human review flagging

### **Information Theory**
- **Context Optimization**: Dynamic window size selection
- **Feature Fusion**: Optimal modality combination
- **Attention Mechanisms**: Information-theoretic weighting

## 🛠️ **Development Setup**

### **Local Development (MacBook M4)**
```bash
# Recommended for development and small experiments
python -m venv senticcrystal
source senticcrystal/bin/activate
pip install -r requirements.txt
```

### **High-Performance Training (Saturn Cloud A100)**
```bash
# For full-scale experiments and production training
conda env create -f docs/setup/environment_saturn_cloud.yml
conda activate senticcrystal-saturn
```

See [`docs/setup/saturn_cloud_setup.md`](docs/setup/saturn_cloud_setup.md) for detailed setup instructions.

## 📖 **Documentation**

### **Experiments & Results**
- [📊 Experimental Plan](docs/experiments/EXPERIMENTAL_PLAN.md)
- [📈 Results Summary](docs/experiments/EXPERIMENTAL_RESULTS_SUMMARY.md)  
- [🔄 Turn Analysis Plan](docs/experiments/COMPREHENSIVE_TURN_ANALYSIS_PLAN.md)

### **Technical Analysis**
- [🔍 Codebase Analysis](docs/analysis/COMPREHENSIVE_CODEBASE_ANALYSIS.md)
- [📊 Data Structure Analysis](docs/analysis/IEMOCAP_4WAY_DATA_ANALYSIS.md)
- [🔧 Refactoring Report](docs/analysis/REFACTORING_COMPLETE.md)

### **Setup & Configuration**
- [☁️ Saturn Cloud Setup](docs/setup/saturn_cloud_setup.md)
- [🐍 Environment Configuration](docs/setup/environment_saturn_cloud.yml)

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture Database
- **Hugging Face**: Transformers and Sentence-Transformers libraries
- **WordNet-Affect**: Emotion lexicon resource
- **Saturn Cloud**: High-performance computing platform

---

**🔮 SenticCrystal - Where Emotion Recognition Meets Bayesian Precision**