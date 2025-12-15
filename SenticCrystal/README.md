# SenticCrystal

**State-of-the-art Emotion Recognition in Conversation with Linguistic Analysis**

Achieving **82.69% (4-way)** and **67.07% (6-way)** on IEMOCAP using strictly causal context.

## Key Findings

| Finding | Effect | p-value |
|---------|--------|---------|
| Context is crucial | **+22%** F1 | <10⁻¹⁵ |
| Hierarchical helps (utterance-only) | +1.9% | 0.007 |
| Context subsumes hierarchy | - | 0.43 |
| SenticNet doesn't help | 0% | - |

## Performance

| Method | 4-way | 6-way | Context Type |
|--------|-------|-------|--------------|
| HCAM (2023) | 81.4% | 64.4% | Bidirectional |
| Mai et al. (2019) | 81.5% | - | Intra-utterance |
| **Ours** | **82.69%** | **67.07%** | **Causal** |

## Installation

```bash
conda env create -f environment.yml
conda activate senticcrystal
```

## Project Structure

```
SenticCrystal/
├── src/                          # Core modules
│   ├── models/                   # Model definitions
│   ├── features/                 # Feature extraction
│   └── utils/                    # Utilities
│
├── scripts/
│   ├── statistical_analysis/     # Statistical tests
│   │   ├── 01_context_effect.py
│   │   ├── 02_flat_vs_hier_utterance.py
│   │   ├── 03_flat_vs_hier_turn.py
│   │   ├── 04_emotion_optimal_k.py
│   │   ├── 05_senticnet_analysis.py
│   │   ├── 06_discourse_markers.py
│   │   └── run_all.py
│   │
│   ├── turn/                     # Turn-level training
│   └── generator/                # Embedding generation
│
├── results/
│   └── STATISTICAL_ANALYSIS_SUMMARY.md
│
└── data/                         # IEMOCAP (not included)
```

## Usage

### Run Statistical Analysis

```bash
cd /path/to/SenticCrystal
python scripts/statistical_analysis/run_all.py
```

### Train Turn-level Model

```bash
python scripts/turn/train_turnlevel_k_sweep_bayesian.py \
    --task 4way \
    --encoder sentence-roberta \
    --seed 42
```

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
