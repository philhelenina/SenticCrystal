# SenticCrystal

**State-of-the-art Emotion Recognition in Conversation with Linguistic Analysis**

Achieving **82.69% (4-way)** and **67.07% (6-way)** weighted F1 on IEMOCAP using strictly causal context (text-only).

## Highlights

- **SOTA Performance**: Outperforms bidirectional models using only causal (past) context
- **Statistical Rigor**: 10 random seeds, paired t-tests, Bonferroni correction
- **Linguistic Analysis**: Emotion-specific discourse marker positioning patterns (exploratory)

## Key Findings

| Finding | Effect | p-value | Status |
|---------|--------|---------|--------|
| Context is crucial | **+22%** F1 | <10⁻¹⁵ | Confirmed |
| Hierarchical helps (utterance-only) | +1.9% | 0.007 | Confirmed |
| Context subsumes hierarchy | - | 0.43 | Confirmed |
| SenticNet doesn't help | 0% | - | Negative |
| Discourse marker patterns | Exploratory | 0.01* | Exploratory |

## Performance Comparison

| Method | 4-way | 6-way | Context Type |
|--------|-------|-------|--------------|
| HCAM (2023) | 81.4% | 64.4% | Bidirectional |
| Mai et al. (2019) | 81.5% | - | Intra-utterance |
| **Ours** | **82.69%** | **67.07%** | **Causal** |

*Using strictly causal context (no future utterances) - applicable to real-time systems*

## Method

- **Encoder**: Sentence-RoBERTa (nli-roberta-base-v2)
- **Context**: K-turn sliding window (K=0~200)
- **Pooling**: Layer averaging (avg_last4) + mean pooling
- **Classifier**: LSTM with Bayesian hyperparameter optimization (Optuna)

## Installation

```bash
conda env create -f environment.yml
conda activate senticcrystal
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Project Structure

```
SenticCrystal/
├── src/                          # Core modules
│   ├── models/                   # Model definitions
│   ├── features/                 # Feature extraction (S-RoBERTa)
│   └── utils/                    # Utilities
│
├── scripts/
│   ├── statistical_analysis/     # Statistical tests (6 scripts)
│   ├── turn/                     # Turn-level training
│   ├── trainer/                  # Utterance-level training
│   └── generator/                # Embedding generation
│
├── results/
│   ├── STATISTICAL_ANALYSIS_SUMMARY.md
│   └── discourse_markers/        # DM position data
│
└── data/                         # IEMOCAP (not included)
```

## Usage

### Run Statistical Analysis

```bash
python scripts/statistical_analysis/run_all.py
```

### Train Turn-level Model

```bash
python scripts/turn/train_turnlevel_k_sweep_bayesian.py \
    --task 4way \
    --encoder sentence-roberta \
    --seed 42
```

### Generate Embeddings

```bash
python scripts/generator/generate_sroberta_npz_4way.py
python scripts/generator/generate_sroberta_hier_npz_4way.py
```

## Roadmap

- [x] Text-only baseline (current)
- [ ] Multimodal integration (audio features)
- [ ] Real-time inference pipeline

## Citation

```bibtex
@article{senticcrystal2024,
  title={Understanding Emotion in Discourse: From Recognition to Generation-Informed Insights},
  author={Anonymous},
  journal={TACL (under review)},
  year={2024}
}
```

## License

MIT License

## Contact

Questions: cheonkamjeong@gmail.com
