# SenticCrystal: Emotion Recognition with Conversational Context

## Overview

IEMOCAP 데이터셋을 활용한 대화 감정 인식 연구. Conversational context의 효과와 hierarchical sentence representation의 필요성을 통계적으로 검증.

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset | IEMOCAP |
| Tasks | 4way (angry, happy, sad, neutral), 6way (+excited, frustrated) |
| Encoder | sentence-roberta, sentence-roberta-hier |
| Seeds | Baseline 5개 (42-46), Turn-level 10개 (42-51) |
| Context | K = 0 (utterance) ~ 200 (turn-level) |
| Statistics | Paired t-test, Bonferroni correction |

---

## Main Findings

### 1. Context Effect: Turn-level >> Utterance-level ✅

| Task | Utterance (K=0) | Turn (K=200) | Δ | p-value |
|------|-----------------|--------------|---|---------|
| 4way | 0.6494 ± 0.007 | 0.7959 ± 0.016 | **+22.6%** | **<1e-15 ✓✓✓** |
| 6way | 0.5235 ± 0.013 | 0.6372 ± 0.015 | **+21.7%** | **<1e-15 ✓✓✓** |

**Conclusion**: Conversational context provides +22% F1 improvement.

---

### 2. Hierarchical vs Flat: Context-Dependent ✅

#### Utterance-level: Hier > Flat

| Task | Flat | Hier | Δ | p-value |
|------|------|------|---|---------|
| 4way | 0.6495 ± 0.002 | 0.6687 ± 0.009 | **+1.9%** | **0.007 ✓✓** |
|      | (min=0.6455, max=0.6521) | (min=0.6527, max=0.6784) | | |

#### Turn-level: Flat ≈ Hier

| Task | Flat | Hier | Δ | p-value |
|------|------|------|---|---------|
| 4way | 0.7959 ± 0.016 | 0.7915 ± 0.016 | +0.4% | 0.427 (n.s.) |
|      | (min=0.7736, max=0.8173) | (min=0.7580, max=0.8115) | | |
| 6way | 0.6372 ± 0.015 | 0.6262 ± 0.014 | +1.1% | 0.078 (n.s.) |
|      | (min=0.6121, max=0.6627) | (min=0.5999, max=0.6455) | | |

**Conclusion**: Hierarchical sentence representation helps at utterance-level, but context subsumes this benefit at turn-level.

---

### 3. Discourse Markers: Emotion-Specific Positioning ✅

| Emotion | n | L/R Ratio | Mean Position |
|---------|---|-----------|---------------|
| Angry | 785 | 1.15 | 0.436 |
| Happy | 1800 | **1.34** | **0.418** |
| **Sad** | 1151 | 1.15 | **0.453** |
| Neutral | 1550 | 1.29 | 0.414 |

#### Statistical Test
- **ANOVA**: F=3.778, **p=0.0101** *
- **Pairwise (Bonferroni α=0.0083)**:
  - Happy vs Sad: **p=0.0050 ✓**
  - Sad vs Neutral: **p=0.0024 ✓**

**Conclusion**: Sad utterances have discourse markers positioned more toward the right periphery, consistent with linguistic theories of subjectivity marking.

#### Right Periphery Analysis (Haselow Theory)

Analyzing stance markers "though" and "however" at utterance-final position (>0.7):

| Emotion | n | Mean Position | At End (>0.7) |
|---------|---|---------------|---------------|
| Angry | 11 | 0.287 | 0% |
| Happy | 29 | 0.488 | 24.1% |
| Sad | 31 | 0.424 | 22.6% |
| **Neutral** | 21 | **0.644** | **52.4%** |

- **Chi-square**: χ²=11.313, **p=0.0102** *
- **Finding**: Neutral utterances use stance markers at final position significantly more than emotional utterances

#### Marker-Specific Position Patterns (Chi-Square Tests)

| Marker | χ² | p-value | Key Pattern |
|--------|-----|---------|-------------|
| **MAYBE** | 28.37 | **<0.0001*** | SAD: 75% LEFT (hedging upfront), ANGRY: 40% RIGHT |
| **THOUGH** | 20.20 | **0.0026** | NEUTRAL: 50% RIGHT (stance), ANGRY: 0% RIGHT |
| **AND** | 32.61 | **<0.0001*** | SAD: 24% LEFT (continuation), ANGRY: 24% RIGHT |
| **SO** | 24.48 | **0.0004*** | SAD: 13% LEFT (lowest), conclusion at end |
| **LIKE** | 25.44 | **0.0003*** | ANGRY: 31% LEFT (filler), SAD: 13% LEFT |
| WELL | 13.90 | 0.0307* | All emotions: ~85% LEFT (turn-taking) |

**Linguistic Interpretation**:
- **SAD**: Hedging (maybe) at utterance start, continuation markers (and) at start
- **ANGRY**: Hedging at end (afterthought), filler (like) at start, no stance modification at end
- **NEUTRAL**: Stance markers (though) at right periphery, balanced hedging

#### LP/RP Functional Analysis (Haselow Framework)

Based on Haselow's periphery theory:
- **LP (Left Periphery, <15%)**: Discourse coherence, turn-taking, topic shifts
- **RP (Right Periphery, >85%)**: Illocutionary modification, stance marking

| Marker | χ² | p-value | Key Finding |
|--------|-----|---------|-------------|
| **AND** | 29.72 | **<0.0001*** | SAD: 21% LP (continuation), ANGRY: 19% RP (afterthought) |
| **MAYBE** | 26.04 | **0.0002*** | SAD: 75% LP (hedge upfront), ANGRY: 33% RP (soften later) |
| **SO** | 22.30 | **0.0011** | SAD: 11% LP (delays conclusion), others ~21% LP |
| **THOUGH** | 19.11 | **0.0040** | NEUTRAL: 45% RP (stance mod), ANGRY: 0% RP |
| **WELL** | 16.83 | **0.0099** | SAD: 5% RP (mitigation), others ~0% RP |
| **LIKE** | 13.69 | **0.0332** | ANGRY: 21% LP (filler), SAD: 11% LP |

**Emotion Profiles**:

| Emotion | LP Pattern | RP Pattern | Interpretation |
|---------|------------|------------|----------------|
| **ANGRY** | LIKE filler (21%), low AND (7%) | MAYBE (33%), AND (19%) | Assertive + afterthought qualifications |
| **SAD** | MAYBE hedging (75%), AND (21%) | SO conclusion (21%), WELL (5%) | Uncertain openings, delayed conclusions |
| **HAPPY** | Balanced SO (21%) | Balanced SO (22%) | Flexible marker placement |
| **NEUTRAL** | WELL turn-taking (86%) | THOUGH stance (45%) | Formal discourse, stance modification |

---

### 4. SenticNet Integration: Negative Result ❌

| Condition | Baseline | +SenticNet | Δ |
|-----------|----------|------------|---|
| 4way | baseline | -0.94% max | negative |
| 6way | baseline | ~0% | no change |

**Conclusion**: Lexical knowledge (SenticNet) provides no additional benefit. Pre-trained embeddings already capture this information.

---

### 5. Emotion-Specific Optimal K: Not Significant ❌

| Emotion (4way) | Optimal K (Mean ± Std) |
|----------------|------------------------|
| Angry | 152.0 ± 29.6 |
| Happy | 130.0 ± 46.9 |
| Sad | 120.0 ± 33.5 |
| Neutral | 102.0 ± 47.3 |

- Pairwise differences: p > 0.05 after Bonferroni correction
- **Conclusion**: No statistically significant difference in optimal context length across emotions.

---

## Summary Table

| Finding | Effect Size | p-value | Status |
|---------|-------------|---------|--------|
| **Context Effect (Turn > Utt)** | **+22%** | **<1e-15** | ✅ Main |
| **Hier > Flat (Utterance)** | **+1.9%** | **0.007** | ✅ Significant |
| **Flat ≈ Hier (Turn)** | +0.4% | 0.43 | ✅ Context subsumes |
| **Discourse Marker Position** | Sad ≠ Happy | **0.005** | ✅ Significant |
| **Right Periphery (Haselow)** | Neutral > Sad | **0.010** | ✅ Significant |
| **Marker-Specific Patterns** | MAYBE, THOUGH, etc. | **<0.001** | ✅ 5 markers sig. |
| SenticNet | 0% | - | ❌ Negative |
| Emotion Optimal K | varies | >0.05 | ❌ Not significant |

---

## Key Contributions

1. **Conversational context is crucial**: +22% F1 improvement with K=200 context turns
2. **Hierarchical representation is context-dependent**: Beneficial only at utterance-level; context subsumes hierarchical structure benefits
3. **Emotion-specific discourse patterns**: Sad emotion shows distinct right-periphery positioning of subjective markers (p=0.005)
4. **LP/RP functional diversification by emotion** (Haselow Framework):
   - **SAD**: Hedging at LP (MAYBE 75%), continuation markers at LP (AND 21%)
   - **ANGRY**: Afterthought qualifications at RP (MAYBE 33%, AND 19%), no stance modification
   - **NEUTRAL**: Formal turn-taking at LP (WELL 86%), stance modification at RP (THOUGH 45%)
5. **Pre-trained embeddings suffice**: External lexical knowledge (SenticNet) provides no additional benefit

---

## Statistical Methods

### Paired t-test
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

### Bonferroni Correction
$$\alpha_{corrected} = \frac{0.05}{m}$$

---

## Data Locations

```
results/
├── analysis/
│   └── all_results_combined.csv          # Utterance-level baseline
├── turnlevel_k_sweep_bayesian/           # Turn-level experiments
│   ├── 4way_sentence-roberta_avg_last4_mean_flat/seed*/
│   └── 4way_sentence-roberta-hier_avg_last4_mean_mean/seed*/
├── senticnet_experiments/
│   └── results.csv                       # SenticNet fusion results
└── discourse_markers/
    └── markers_4way_extracted.csv        # Discourse marker positions
```

---

## Scripts

```
scripts/statistical_analysis/
├── 01_context_effect.py           # Turn vs Utterance
├── 02_flat_vs_hier_utterance.py   # Baseline Flat vs Hier
├── 03_flat_vs_hier_turn.py        # Turn-level Flat vs Hier
├── 04_emotion_optimal_k.py        # Emotion-specific K
├── 05_senticnet_analysis.py       # SenticNet analysis
├── 06_discourse_markers.py        # Discourse marker analysis
└── run_all.py                     # Run all analyses
```
