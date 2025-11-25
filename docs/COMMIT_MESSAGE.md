# Commit Message Template

## Short Version (for git commit -m)

```
feat: IEMOCAP emotion recognition pipeline with hierarchical Sentence-RoBERTa

- Flat & hierarchical embedding generation (4-way/6-way)
- Multiple layer combination strategies (last, avg_last4, scalar-weighted)
- 9 token pooling methods including positional & IDF weighting
- 5 sentence aggregators for hierarchical models
- MLP/LSTM classifiers with class-weighted loss
- Multi-GPU experiment runners (n=10 seeds, 1200 total runs)
```

## Detailed Version

```
feat: Comprehensive IEMOCAP emotion recognition experimental framework

This commit introduces a systematic ablation study for Emotion Recognition
in Conversation (ERC) using Sentence-RoBERTa embeddings on IEMOCAP.

EMBEDDING GENERATION:
- Flat (2D): Single utterance → [N, 768] embeddings
- Hierarchical (3D): Sentence-level → [N, S_max, 768] with length tracking

LAYER COMBINATION STRATEGIES:
- last: Final transformer layer (task-specific features)
- avg_last4: Average of layers 9-12 (multi-level abstraction)
- last4_scalar_*: Learned-inspired presets (up/down/top2 weighting)

TOKEN POOLING METHODS:
- Standard: cls, mean, attention-weighted
- Positional: wmean_pos (front), wmean_pos_rev (end emphasis)
- Temporal: wmean_exp_fast/med/slow (exponential decay variants)
- Lexical: wmean_idf (inverse document frequency weighting)

HIERARCHICAL AGGREGATION:
- Simple: mean, sum (baseline approaches)
- Weighted: expdecay (recency bias for final emotional state)
- Learned: attn (salience-based), lstm (sequential dynamics)

CLASSIFICATION:
- MLP: Fast, interpretable baseline
- LSTM: Sequential pattern modeling

EXPERIMENTAL DESIGN:
- Tasks: 4-way (ang/hap/sad/neu) and 6-way (+exc/fru)
- Seeds: 42-51 (n=10 for statistical significance)
- GPU parallelization: 4-way distribution for efficiency
- Class-weighted CrossEntropy for imbalance handling

TOTAL: 1,200 experimental runs (720 flat + 480 hierarchical)
```

## Semantic Commit Format

```
feat(erc): add IEMOCAP emotion classification pipeline

BREAKING CHANGE: None

Refs: SenticCrystal project
```
