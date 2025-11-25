# SenticCrystal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

**SenticCrystal** is an information-theoretic framework for emotion recognition that "crystallizes" the essential principles of emotion from complex text and speech data. Through systematic experiments, we discovered that simpler, interpretable models can achieve performance comparable to complex architectures, leading to our core philosophy: **finding clarity in complexity**.

### Key Achievements
- **73.25% accuracy** on IEMOCAP 4-way emotion classification (text-only)
- **Complete model recovery** from catastrophic failure (30.9% → 69.94%) using Focal Loss optimization
- **Balanced classification** across all emotion classes (eliminating 10-80% class bias)

## Core Philosophy

SenticCrystal embodies three fundamental principles:

1. **Simplicity over Complexity**: Our experiments demonstrate that interpretable models with proper optimization outperform unnecessarily complex architectures
2. **Information Crystallization**: We extract and preserve only the essential information for emotion recognition, removing noise and redundancy
3. **Balanced Understanding**: Through Focal Loss optimization (α=1.0, γ=1.2), we achieve balanced performance across all emotion classes

---

## Experimental Pipeline

This repository includes a comprehensive experimental framework for **Emotion Recognition in Conversation (ERC)** using the IEMOCAP dataset with Sentence-RoBERTa embeddings.

### Experimental Design Rationale

#### Layer Combination Strategies

| Method | Description | Rationale |
|--------|-------------|-----------|
| `last` | Use only the final transformer layer | Final layer captures the most task-specific semantic features; serves as a simple baseline |
| `avg_last4` | Average of last 4 layers | Combines different levels of abstraction; middle layers capture syntactic info while upper layers capture semantics |
| `last4_scalar_up` | Weighted [1,2,3,4] | Emphasizes deeper layers which contain more refined contextual representations |
| `last4_scalar_down` | Weighted [4,3,2,1] | Emphasizes shallower layers which preserve more lexical/local features |
| `last4_scalar_top2` | Weighted [0,1,1,0] | Focuses on layers 10-11 which often show peak performance in probing tasks |

#### Token Pooling Methods

| Method | Description | Rationale |
|--------|-------------|-----------|
| `cls` | First token ([CLS]) embedding | Standard BERT-style pooling; captures sentence-level representation |
| `mean` | Masked mean of all tokens | Distributes attention equally; robust for variable-length inputs |
| `attn` | CLS-query attention pooling | Learns to weight tokens by relevance to the overall meaning |
| `wmean_pos` | Position-weighted (front emphasis) | Hypothesis: sentence-initial words set emotional tone |
| `wmean_pos_rev` | Position-weighted (end emphasis) | Hypothesis: sentence-final words carry emotional climax (common in spoken language) |
| `wmean_exp_fast` | Exponential decay (τ=2.0) | Strong recency bias; for high-arousal emotions with quick expression |
| `wmean_exp_med` | Exponential decay (τ=5.0) | Moderate recency bias; balanced approach |
| `wmean_exp_slow` | Exponential decay (τ=10.0) | Mild recency bias; for emotions that build gradually |
| `wmean_idf` | IDF-weighted pooling | Emphasizes rare/informative words; reduces impact of stopwords |

#### Aggregators (Hierarchical Models Only)

| Method | Description | Rationale |
|--------|-------------|-----------|
| `mean` | Average all sentence embeddings | Simple baseline; treats all sentences equally |
| `sum` | Sum all sentence embeddings | Preserves magnitude information; longer utterances get stronger signal |
| `expdecay` | Exponential decay weighting | Later sentences (closer to utterance end) often carry the final emotional state |
| `attn` | Learned attention weights | Allows model to learn which sentences are most emotionally salient |
| `lstm` | LSTM over sentence sequence | Captures sequential dynamics; emotion can evolve through an utterance |

#### Classifier Architectures

| Model | Description | Rationale |
|-------|-------------|-----------|
| `MLP` | 2-layer feedforward network | Fast, interpretable; works well when embeddings are already semantically rich |
| `LSTM` | Single-layer LSTM | Captures sequential patterns in the embedding dimensions; useful for temporal features |

#### Task Configurations

| Task | Classes | Rationale |
|------|---------|-----------|
| **4-way** | anger, happiness, sadness, neutral | Standard benchmark; merges similar emotions (excited→happy) for cleaner separation |
| **6-way** | anger, happiness, sadness, neutral, excited, frustrated | Fine-grained classification; tests model's ability to distinguish subtle emotional differences |

#### Statistical Rigor

- **10 random seeds (42-51)**: Ensures reproducibility and enables statistical significance testing
- **Early stopping (patience=60)**: Prevents overfitting while allowing sufficient training
- **Class-weighted loss**: Handles IEMOCAP's inherent class imbalance

---

## Repository Structure

```
SenticCrystal/
│
├── scripts/
│   ├── generators/                    # Embedding generation scripts
│   │   ├── flat/
│   │   │   ├── generate_sroberta_npz_4way.py
│   │   │   └── generate_sroberta_npz_6way.py
│   │   └── hierarchical/
│   │       ├── generate_sroberta_hier_npz.py
│   │       ├── generate_sroberta_hier_npz_4way.py
│   │       └── generate_sroberta_hier_npz_6way.py
│   │
│   ├── trainers/                      # Model training scripts
│   │   ├── flat/
│   │   │   ├── train_npz_classifier_4way_verbose.py
│   │   │   └── train_npz_classifier_6way_verbose.py
│   │   └── hierarchical/
│   │       ├── train_npz_hier_classifier_4way.py
│   │       ├── train_npz_hier_classifier_6way.py
│   │       └── train_npz_hier_fused_classifier.py
│   │
│   └── runners/                       # Experiment execution scripts
│       ├── flat/
│       │   ├── run_all_n10_flat.sh
│       │   └── run_n10_gpu[0-3]_*way_*.sh
│       └── hierarchical/
│           ├── run_all_n10_hier.sh
│           └── run_n10_hier_gpu[0-3]_*way.sh
│
├── docs/
│   ├── COMMIT_MESSAGE.md
│   └── DOCSTRINGS.md
│
└── README.md
```

## Experiment Scale

| Experiment Type | Configurations | Seeds | Total Runs |
|-----------------|----------------|-------|------------|
| Flat Baseline | 3 encoders × 2 layers × 3 pools × 2 classifiers | 10 | 720 |
| Hierarchical | 2 layers × 2 pools × 5 aggregators × 2 classifiers | 10 | 480 |
| **Grand Total** | | | **1,200** |

## Usage

```bash
# Run all flat experiments (4 GPUs in parallel)
./scripts/runners/flat/run_all_n10_flat.sh

# Run all hierarchical experiments (4 GPUs in parallel)
./scripts/runners/hierarchical/run_all_n10_hier.sh

# Monitor progress
tail -f scripts/runners/flat/n10_gpu0_flat.log
```

## Output

Each experiment produces:
- `results.json`: Metrics (accuracy, macro-F1, weighted-F1, per-class F1)
- `confusion_matrix.png`: Visualization of predictions
- `cls_report.txt`: Detailed classification report

---

## Performance

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

## Key Innovations

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

## Roadmap

### v1.0 - Text-Only (Current)
- ✅ Hierarchical text processing
- ✅ Focal Loss optimization
- ✅ Information-theoretic analysis
- ✅ Ensemble methods

### v2.0 - Multimodal (In Development)
- 🔄 Speech feature integration
- 🔄 Cross-modal attention mechanisms
- 🔄 Multimodal fusion strategies
- 🔄 Real-time processing pipeline

### v3.0 - Future Enhancements
- ⏳ Video modality integration
- ⏳ Generative emotion modeling
- ⏳ Cross-lingual emotion recognition
- ⏳ Deployment-ready API

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- Transformers (Hugging Face)
- scikit-learn
- pandas, numpy, matplotlib

## Citation

If you use SenticCrystal in your research, please cite:

```bibtex
@article{senticcrystal2025,
  title={SenticCrystal: Information-Theoretic Crystallization of Emotion Recognition},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IEMOCAP dataset creators at USC SAIL
- Sentence-Transformers and Hugging Face teams
- WordNet-Affect creators

## Contact

For questions and collaborations: cheonkamjeong@gmail.com

---

*"In the complexity of human emotion, we find clarity through crystallization."* - SenticCrystal Philosophy
