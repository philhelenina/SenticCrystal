# SenticCrystal Statistical Analysis Suite

## Overview

IEMOCAP 감정 인식 실험에 대한 통계 분석 스크립트 모음입니다.

## Directory Structure

```
SenticCrystal/
├── results/
│   ├── analysis/
│   │   ├── all_results_combined.csv      # Baseline (Utterance-level) 결과
│   │   ├── summary_combined.csv          # 요약 통계
│   │   └── flat_vs_hierarchical.csv      # Flat vs Hier 비교
│   │
│   ├── turnlevel_k_sweep_bayesian/       # Turn-level 실험 결과
│   │   ├── 4way_sentence-roberta_avg_last4_mean_flat/
│   │   │   ├── seed42/
│   │   │   │   ├── Ks.npy                # K 값 배열 [0, 10, 20, ..., 200]
│   │   │   │   ├── preds_perK.npy        # 예측 확률 (num_K, N, C)
│   │   │   │   ├── labels.npy            # 정답 레이블 (N,)
│   │   │   │   ├── k_sweep_results.csv   # K별 성능
│   │   │   │   └── metadata.json         # 실험 설정
│   │   │   ├── seed43/
│   │   │   └── ...                       # seed42-51 (10 seeds)
│   │   │
│   │   ├── 4way_sentence-roberta-hier_avg_last4_mean_mean/  # Hier 버전
│   │   ├── 6way_sentence-roberta_avg_last4_mean_flat/
│   │   └── 6way_sentence-roberta-hier_avg_last4_mean_mean/
│   │
│   ├── senticnet_experiments/
│   │   ├── results.csv                   # SenticNet fusion 결과
│   │   └── analysis/
│   │       └── comparison_summary.csv    # Baseline 대비 비교
│   │
│   ├── discourse_markers/
│   │   ├── markers_4way_extracted.csv    # 담화표지 추출 결과
│   │   └── markers_6way_extracted.csv
│   │
│   └── STATISTICAL_ANALYSIS_SUMMARY.md   # 최종 분석 요약
│
└── scripts/
    └── statistical_analysis/
        ├── README.md                     # 이 파일
        ├── run_all.py                    # 전체 실행 스크립트
        ├── 01_context_effect.py          # Context 효과 분석
        ├── 02_flat_vs_hier_utterance.py  # Utterance-level Flat vs Hier
        ├── 03_flat_vs_hier_turn.py       # Turn-level Flat vs Hier
        ├── 04_emotion_optimal_k.py       # 감정별 optimal K 분석
        ├── 05_senticnet_analysis.py      # SenticNet 분석
        └── 06_discourse_markers.py       # 담화표지 분석
```

## Analysis Scripts

### 01_context_effect.py
**목적**: Turn-level (K=200) vs Utterance-level (K=0) 비교

**입력 데이터**:
- `results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*/`

**통계 방법**:
- Paired t-test (같은 seed에서 K=0 vs K=200)

**결과**:
| Task | Utterance | Turn | Δ | p-value |
|------|-----------|------|---|---------|
| 4way | 0.6494 | 0.7959 | **+22.6%** | **<1e-15** |
| 6way | 0.5235 | 0.6372 | **+21.7%** | **<1e-15** |

---

### 02_flat_vs_hier_utterance.py
**목적**: Baseline (Utterance-level)에서 Flat vs Hierarchical 비교

**입력 데이터**:
- `results/analysis/all_results_combined.csv`

**통계 방법**:
- Paired t-test (같은 seed에서 Flat vs Hier)

**결과**:
| Task | Flat | Hier | Δ | p-value |
|------|------|------|---|---------|
| 4way | 0.6495 | 0.6687 | **+1.9%** | **0.007** |

---

### 03_flat_vs_hier_turn.py
**목적**: Turn-level에서 Flat vs Hierarchical 비교 (모든 K값)

**입력 데이터**:
- `results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*/`
- `results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta-hier_avg_last4_mean_mean/seed*/`

**통계 방법**:
- Paired t-test at each K
- Bonferroni correction (21 comparisons, α = 0.0024)

**결과**:
- 대부분 K에서 Flat ≈ Hier (n.s.)
- 6way K=170: Flat > Hier (p=0.0009, Bonferroni significant)

---

### 04_emotion_optimal_k.py
**목적**: 감정별 최적 K값 분석 및 감정 간 차이 비교

**입력 데이터**:
- `results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*/`

**통계 방법**:
- Per-seed optimal K 계산
- Pairwise paired t-test with Bonferroni correction

**결과**:
- 감정별 optimal K 차이는 통계적으로 유의하지 않음 (Bonferroni 보정 후)

---

### 05_senticnet_analysis.py
**목적**: SenticNet lexical feature fusion 효과 분석

**입력 데이터**:
- `results/senticnet_experiments/results.csv`

**결과**:
- **Negative result**: SenticNet이 개선을 주지 않음
- Pre-trained embeddings가 이미 lexical knowledge를 포함

---

### 06_discourse_markers.py
**목적**: 발화 내 담화표지 위치 분석

**입력 데이터**:
- `results/discourse_markers/markers_{task}_extracted.csv`

**결과**:
- L/R ratio = 2.18 (descriptive)
- wmean_pos_rev vs mean: 유의하지 않음

---

## Usage

```bash
cd /home/cheonkaj/projects/SenticCrystal

# Run all analyses
python scripts/statistical_analysis/run_all.py

# Or run individually
python scripts/statistical_analysis/01_context_effect.py
python scripts/statistical_analysis/02_flat_vs_hier_utterance.py
# ...
```

## Key Findings Summary

| Finding | Effect Size | p-value | Significance |
|---------|-------------|---------|--------------|
| **Context Effect** | **+22%** | **<1e-15** | ✅ Main |
| **Hier > Flat (Utt)** | **+1.9%** | **0.007** | ✅ Significant |
| Flat ≈ Hier (Turn) | +0.4% | 0.43 | n.s. |
| Emotion optimal K | varies | >0.05 | n.s. |
| SenticNet | 0% | - | Negative |

## Statistical Methods

### Paired t-test
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

- $\bar{d}$: 쌍별 차이의 평균
- $s_d$: 쌍별 차이의 표준편차
- $n$: seed 수

### Bonferroni Correction
$$\alpha_{corrected} = \frac{\alpha}{m}$$

- $\alpha = 0.05$
- $m$: 비교 횟수

## References

- Dataset: IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- Tasks: 4way (angry, happy, sad, neutral), 6way (+excited, frustrated)
- Encoder: sentence-roberta, sentence-roberta-hier
