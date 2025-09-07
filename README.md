# SenticCrystal
An information-theoretic framework for crystallizing the core principles of emotion from text and speech.

# SenticCrystal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📌 Overview

**SenticCrystal** is an information-theoretic framework for emotion recognition that "crystallizes" the essential principles of emotion from complex text and speech data. Through systematic experiments, we discovered that simpler, interpretable models can achieve performance comparable to complex architectures, leading to our core philosophy: **finding clarity in complexity**.

### Key Achievements
- **73.25% accuracy** on IEMOCAP 4-way emotion classification (text-only)
- **Complete model recovery** from catastrophic failure (30.9% → 69.94%) using Focal Loss optimization
- **Balanced classification** across all emotion classes (eliminating 10-80% class bias)

## 🎯 Core Philosophy

SenticCrystal embodies three fundamental principles:

1. **Simplicity over Complexity**: Our experiments demonstrate that interpretable models with proper optimization outperform unnecessarily complex architectures
2. **Information Crystallization**: We extract and preserve only the essential information for emotion recognition, removing noise and redundancy
3. **Balanced Understanding**: Through Focal Loss optimization (α=1.0, γ=1.2), we achieve balanced performance across all emotion classes

## 🏗️ Project Structure

```
SenticCrystal/
│
├── src/                          # Core reusable components
│   ├── models/                   # Model architectures
│   │   ├── mlp.py               # MLP classifier with Focal Loss
│   │   ├── lstm_context.py      # Contextual LSTM implementations
│   │   └── ensemble.py          # Ensemble methods
│   │
│   ├── features/                 # Feature extraction modules
│   │   ├── wordnet_affect.py    # WordNet-Affect emotional embeddings
│   │   ├── sentence_roberta.py  # Sentence-level RoBERTa embeddings
│   │   └── context_window.py    # Multi-turn context processing
│   │
│   ├── analysis/                 # Analysis tools
│   │   ├── information_theory.py # Entropy, MI calculations
│   │   ├── class_balance.py     # Class imbalance analysis
│   │   └── confidence_metrics.py # Prediction confidence analysis
│   │
│   └── utils/                    # Utility functions
│       ├── data_loader.py       # IEMOCAP data loading
│       ├── preprocessing.py     # Text preprocessing
│       └── focal_loss.py        # Focal Loss implementation
│
├── scripts/                      # Execution scripts (workflow-ordered)
│   ├── 1_data_preparation/
│   │   ├── prepare_iemocap.py   # IEMOCAP dataset preparation
│   │   └── generate_embeddings.py # Generate text embeddings
│   │
│   ├── 2_training/
│   │   ├── train_baseline.py    # Train baseline models
│   │   ├── train_focal_loss.py  # Train with Focal Loss
│   │   └── train_ensemble.py    # Train ensemble models
│   │
│   ├── 3_evaluation/
│   │   ├── evaluate_models.py   # Model evaluation
│   │   ├── analyze_errors.py    # Error analysis
│   │   └── generate_reports.py  # Generate performance reports
│   │
│   └── 4_experiments/
│       ├── ablation_study.py    # Component ablation studies
│       └── parameter_search.py  # Hyperparameter optimization
│
├── configs/                      # Configuration files
│   ├── model_configs.yaml       # Model configurations
│   ├── training_configs.yaml    # Training parameters
│   └── focal_loss_params.yaml   # Optimal Focal Loss parameters
│
├── results/                      # Outputs and results
│   ├── models/                  # Saved model checkpoints
│   ├── logs/                    # Training logs
│   ├── figures/                 # Visualizations
│   └── reports/                 # Performance reports
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb   # Data analysis
│   └── result_visualization.ipynb # Result visualization
│
├── tests/                        # Unit tests
│   └── test_focal_loss.py       # Test Focal Loss implementation
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- IEMOCAP dataset access (requires license agreement)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SenticCrystal.git
cd SenticCrystal
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models and resources:
```bash
python scripts/download_resources.py
```

### Quick Start

1. **Prepare IEMOCAP dataset:**
```bash
python scripts/1_data_preparation/prepare_iemocap.py \
    --data_path /path/to/IEMOCAP \
    --output_path data/processed/
```

2. **Generate embeddings:**
```bash
python scripts/1_data_preparation/generate_embeddings.py \
    --config configs/model_configs.yaml \
    --context_window 5
```

3. **Train model with Focal Loss:**
```bash
python scripts/2_training/train_focal_loss.py \
    --alpha 1.0 \
    --gamma 1.2 \
    --config configs/training_configs.yaml
```

4. **Evaluate performance:**
```bash
python scripts/3_evaluation/evaluate_models.py \
    --model_path results/models/best_model.pth \
    --test_data data/processed/test.pkl
```

## 📊 Performance

### Text-Only Model (v1.0)

| Configuration | Baseline | With Focal Loss | Improvement |
|--------------|----------|-----------------|-------------|
| Config 146 (RoBERTa) | 72.1% | 70.75% | Balanced* |
| Config 1 (WN+RoBERTa) | 65.8% | 70.10% | +4.30% |
| Config 2 (Context LSTM) | 30.9% | 69.94% | +39.04% |
| **Ensemble (Weighted)** | - | **71.56%** | - |

*Note: While Config 146 shows slight accuracy decrease, it achieves significantly better class balance

### Per-Class Performance (After Focal Loss)

| Emotion | Before FL | After FL | Improvement |
|---------|-----------|----------|-------------|
| Angry | 14.3% | 68% | +53.7% |
| Happy | 47.8% | 77% | +29.2% |
| Sad | 61.4% | 71% | +9.6% |
| Neutral | 10.0% | 66% | +56.0% |

## 🔬 Key Innovations

1. **Focal Loss Optimization for Emotions**
   - Discovered optimal parameters: α=1.0, γ=1.2 (vs. standard γ=2.0)
   - Moderate focusing better suited for emotion recognition

2. **Information-Theoretic Diagnosis**
   - Shannon entropy for uncertainty quantification
   - Mutual information for feature importance
   - Context dependency analysis by emotion type

3. **Failed Model Recovery**
   - First demonstration of complete recovery from catastrophic failure
   - 30.9% → 69.94% accuracy through systematic optimization

## 🗺️ Roadmap

### v1.0 - Text-Only (Current)
- ✅ Hierarchical text processing
- ✅ Focal Loss optimization
- ✅ Information-theoretic analysis
- ✅ Ensemble methods

### v2.0 - Multimodal (In Development)
- 🔄 Speech feature integration (Emotion2Vec)
- 🔄 Cross-modal attention mechanisms
- 🔄 Multimodal fusion strategies
- 🔄 Real-time processing pipeline

### v3.0 - Future Enhancements
- ⏳ Video modality integration
- ⏳ Generative emotion modeling
- ⏳ Cross-lingual emotion recognition
- ⏳ Deployment-ready API

## 📝 Citation

If you use SenticCrystal in your research, please cite:

```bibtex
@article{senticcrystal2025,
  title={SenticCrystal: Information-Theoretic Crystallization of Emotion Recognition},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IEMOCAP dataset creators at USC SAIL
- Sentence-Transformers and Hugging Face teams
- WordNet-Affect creators

## 📧 Contact

For questions and collaborations: cheonkamjeong@gmail.com

---

*"In the complexity of human emotion, we find clarity through crystallization."* - SenticCrystal Philosophy
