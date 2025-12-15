# SenticCrystal 실험 결과 분석 (n=10 seeds)

## 📊 전체 데이터 요약

- **총 실험 수**: 1,520 experiments
- **Configuration 수**: 152 unique configurations
- **Seeds**: 42-51 (10 seeds)
- **Tasks**: 4-way, 6-way classification

---

## 🏆 최고 성능 Configuration

### 4-Way Classification

#### Flat Architecture (Best)
- **Encoder**: sentence-roberta
- **Layer**: last
- **Pool**: mean
- **Classifier**: lstm
- **Weighted F1**: **0.6517 ± 0.0116**
- **Macro F1**: 0.6435 ± 0.0129
- **Accuracy**: 0.6495 ± 0.0117

#### Hierarchical Architecture (Best)
- **Encoder**: sentence-roberta-hier
- **Layer**: avg_last4
- **Pool**: wmean_pos_rev
- **Aggregator**: mean
- **Classifier**: mlp
- **Weighted F1**: **0.6846 ± 0.0109** ⭐
- **Macro F1**: 0.6783 ± 0.0116
- **Accuracy**: 0.6836 ± 0.0107

**📈 Performance Gain: +5.04% (Hierarchical > Flat)**

---

### 6-Way Classification

#### Flat Architecture (Best)
- **Encoder**: sentence-roberta
- **Layer**: avg_last4
- **Pool**: mean
- **Classifier**: lstm
- **Weighted F1**: **0.5269 ± 0.0104**
- **Macro F1**: 0.5147 ± 0.0115
- **Accuracy**: 0.5294 ± 0.0108

#### Hierarchical Architecture (Best)
- **Encoder**: sentence-roberta-hier
- **Layer**: last
- **Pool**: wmean_pos_rev
- **Aggregator**: attn
- **Classifier**: lstm
- **Weighted F1**: **0.5424 ± 0.0119** ⭐
- **Macro F1**: 0.5254 ± 0.0113
- **Accuracy**: 0.5415 ± 0.0123

**📈 Performance Gain: +2.95% (Hierarchical > Flat)**

---

## 🧪 통계 테스트 결과

### ✅ TEST 1: Flat vs Hierarchical (Main Hypothesis)

#### 4-Way Classification
```
Hierarchical: 0.6846 ± 0.0104
Flat:         0.6517 ± 0.0110
```
- **Mann-Whitney U Test**: p = 0.000220 ***
- **Effect Size (Cohen's d)**: 2.9194 (large)
- **Improvement**: +3.29% absolute, +5.04% relative
- **✅ 결론**: Hierarchical이 Flat보다 **통계적으로 매우 유의하게** 우수함

#### 6-Way Classification
```
Hierarchical: 0.5424 ± 0.0119
Flat:         0.5269 ± 0.0104
```
- **Mann-Whitney U Test**: p = 0.000116 ***
- **Effect Size (Cohen's d)**: 2.4086 (large)
- **Improvement**: +1.55% absolute, +2.95% relative
- **✅ 결론**: Hierarchical이 Flat보다 **통계적으로 매우 유의하게** 우수함

---

### ✅ TEST 2: Encoder Comparison (Flat Only)

#### 4-Way Classification
```
sentence-roberta: 0.6463 ± 0.0104
roberta-base:     0.6348 ± 0.0155
bert-base:        0.6213 ± 0.0157
```

**Pairwise Comparisons:**
1. **sentence-roberta vs bert-base**
   - p < 0.001 ***, Cohen's d = 1.8682 (large)
   - Improvement: +4.03%

2. **sentence-roberta vs roberta-base**
   - p < 0.001 ***, Cohen's d = 0.8706 (large)
   - Improvement: +1.82%

3. **roberta-base vs bert-base**
   - p < 0.001 ***, Cohen's d = 0.8587 (large)
   - Improvement: +2.17%

**✅ 결론**: sentence-roberta > roberta-base > bert-base (모두 통계적으로 유의)

#### 6-Way Classification
```
sentence-roberta: 0.5206 ± 0.0145
roberta-base:     0.4946 ± 0.0190
bert-base:        0.4783 ± 0.0240
```
- 동일한 패턴 확인 (모두 p < 0.001)

---

### ✅ TEST 3: Aggregator Comparison (Hierarchical Only)

#### 4-Way Classification
```
mean:     0.6711 ± 0.0138
attn:     0.6702 ± 0.0112
expdecay: 0.6698 ± 0.0150
sum:      0.6681 ± 0.0124
lstm:     0.6522 ± 0.0205
```

**Key Findings:**
1. **mean vs lstm**: p < 0.001 ***, d = 1.0709 (large), +2.89%
2. **attn vs lstm**: p < 0.001 ***, d = 1.0811 (large), +2.75%
3. **mean vs attn**: p = 0.643 (not significant)
4. **mean vs sum**: p = 0.058 (marginal)

**✅ 결론**: mean, attn, expdecay, sum 모두 비슷한 성능 (유의한 차이 없음)
            lstm aggregator는 유의하게 낮은 성능

#### 6-Way Classification
```
attn:     0.5326 ± 0.0121 ⭐
mean:     0.5287 ± 0.0132
sum:      0.5286 ± 0.0131
expdecay: 0.5270 ± 0.0140
lstm:     0.5149 ± 0.0220
```

**Key Findings:**
1. **attn vs lstm**: p < 0.001 ***, d = 0.9899 (large), +3.43%
2. **attn vs expdecay**: p = 0.008 **, d = 0.4287 (small), +1.07%
3. **attn vs sum**: p = 0.015 *, d = 0.3163 (small), +0.76%
4. **attn vs mean**: p = 0.036 * (marginal)

**✅ 결론**: attn이 6-way에서는 best aggregator (mean과는 근소한 차이)

---

### ✅ TEST 4: Classifier Comparison (MLP vs LSTM)

#### 4-Way Classification
**Flat:**
- MLP: 0.6356 ± 0.0164
- LSTM: 0.6327 ± 0.0182
- p = 0.092 (not significant), d = 0.1675

**Hierarchical:**
- MLP: 0.6660 ± 0.0170
- LSTM: 0.6665 ± 0.0161
- p = 0.701 (not significant), d = 0.0298

**✅ 결론**: MLP와 LSTM 간 유의한 차이 없음 (4-way)

#### 6-Way Classification
**Flat:**
- MLP: 0.5018 ± 0.0206
- LSTM: 0.4938 ± 0.0266
- p = 0.006 **, d = 0.3335 (small), +1.61%

**Hierarchical:**
- MLP: 0.5264 ± 0.0156
- LSTM: 0.5264 ± 0.0172
- p = 0.992 (not significant)

**✅ 결론**: Flat에서는 MLP가 약간 우수, Hierarchical에서는 차이 없음

---

### ✅ TEST 5: Layer Selection (last vs avg_last4)

#### 4-Way Classification
**Flat:**
- avg_last4: 0.6343 ± 0.0157
- last: 0.6369 ± 0.0172
- p = 0.200 (not significant)

**Hierarchical:**
- avg_last4: 0.6689 ± 0.0162
- last: 0.6636 ± 0.0176
- p = 0.007 **, d = 0.3125 (small), +0.80%

**✅ 결론**: Hierarchical에서는 avg_last4가 약간 우수

#### 6-Way Classification
**Flat:**
- avg_last4: 0.4997 ± 0.0231
- last: 0.4959 ± 0.0249
- p = 0.161 (not significant)

**Hierarchical:**
- avg_last4: 0.5231 ± 0.0171
- last: 0.5296 ± 0.0150
- p = 0.001 **, d = 0.4000 (small), last가 +1.22% 우수!

**✅ 결론**: 6-way hierarchical에서는 last가 더 우수

---

### ✅ TEST 6: Pooling Strategy

#### 4-Way Classification
**Flat:**
- mean: 0.6375 ± 0.0161
- wmean_pos_rev: 0.6351 ± 0.0166
- attn: 0.6302 ± 0.0164
- wmean_pos_rev vs mean: p = 0.231 (not significant)

**Hierarchical:**
- wmean_pos_rev: 0.6700 ± 0.0151
- mean: 0.6683 ± 0.0166
- p = 0.378 (not significant)

**✅ 결론**: mean과 wmean_pos_rev 간 유의한 차이 없음

#### 6-Way Classification
**Flat:**
- mean: 0.5032 ± 0.0226
- wmean_pos_rev: 0.5005 ± 0.0226
- attn: 0.4897 ± 0.0249
- attn vs mean: p < 0.001 ***, d = 0.5659 (medium), -2.68%

**Hierarchical:**
- mean: 0.5279 ± 0.0153
- wmean_pos_rev: 0.5249 ± 0.0174
- p = 0.098 (not significant)

**✅ 결론**: mean이 가장 안정적, attn pooling은 6-way flat에서 성능 저하

---

## 📈 주요 발견 (Key Findings)

### 1. 아키텍처 비교
✅ **Hierarchical architecture가 Flat보다 명확히 우수**
- 4-way: +5.04% improvement (p < 0.001, d = 2.92)
- 6-way: +2.95% improvement (p < 0.001, d = 2.41)
- Large effect size로 실질적으로도 의미 있는 개선

### 2. Encoder 선택
✅ **sentence-roberta가 최고 성능**
- sentence-roberta > roberta-base > bert-base
- 모든 비교에서 통계적으로 유의한 차이 (p < 0.001)
- 4-way: sentence-roberta가 bert-base 대비 +4.03%

### 3. Aggregator 선택 (Hierarchical)
✅ **mean, attn, sum, expdecay 모두 비슷한 성능**
- 4-way: mean, attn 권장 (lstm은 제외)
- 6-way: attn이 slightly better (p < 0.05)
- lstm aggregator는 성능이 유의하게 낮음 (제외 권장)

### 4. Classifier 선택
✅ **MLP와 LSTM 간 큰 차이 없음**
- 대부분 경우 유의한 차이 없음
- 6-way flat에서만 MLP가 약간 우수 (p < 0.01)

### 5. Layer Selection
⚠️ **Task와 architecture에 따라 다름**
- 4-way hierarchical: avg_last4가 약간 우수
- 6-way hierarchical: last가 약간 우수
- Effect size가 small이므로 큰 차이는 아님

### 6. Pooling Strategy
✅ **mean pooling이 가장 안정적**
- wmean_pos_rev도 비슷한 성능
- attn pooling은 6-way flat에서 성능 저하

---

## 📊 Seed 간 분산 (Variance across seeds)

```
Task  Type          Acc Std   Macro F1 Std   Weighted F1 Std
4way  flat          0.0110    0.0122         0.0108
4way  hierarchical  0.0140    0.0140         0.0141
6way  flat          0.0112    0.0156         0.0127
6way  hierarchical  0.0148    0.0136         0.0138
```

**분석:**
- Hierarchical이 Flat보다 약간 높은 variance (0.014 vs 0.011)
- 그러나 여전히 낮은 수준 (~1.4%)
- 10 seeds로 안정적인 평가 가능

---

## 🎯 논문을 위한 권장 사항

### 1. Main Claims (강력한 증거)
✅ **Hierarchical architecture가 우수하다**
- p < 0.001, large effect size (d > 2.4)
- 4-way: 68.46% vs 65.17% (+5.04%)
- 6-way: 54.24% vs 52.69% (+2.95%)

### 2. Supporting Claims (강한 증거)
✅ **sentence-roberta가 최고 encoder**
- p < 0.001, large effect size
- 모든 task에서 일관된 우수성

✅ **mean/attn aggregator 권장**
- lstm aggregator 제외 (유의하게 낮은 성능)

### 3. Interesting Observations (약한 증거)
⚠️ **Layer selection은 task-dependent**
- Small effect size
- 추가 분석 필요

⚠️ **MLP vs LSTM은 비슷**
- 대부분 유의한 차이 없음

---

## 📋 필요한 추가 통계 테스트

### 1. Multiple Comparison Correction
현재 많은 pairwise comparison을 수행했으므로:
- **Bonferroni correction** 적용 권장
- **Holm-Bonferroni** 또는 **FDR correction** 고려

### 2. Effect Size Confidence Intervals
- Cohen's d의 95% CI 계산
- Bootstrap 방법 사용

### 3. Best Configuration Validation
- 최고 configuration에 대해 추가 seed (52-61)로 검증
- Generalization 확인

### 4. Task Difficulty Analysis
- 4-way vs 6-way 성능 차이 분석
- 클래스 불균형 영향 분석

### 5. Ablation Study
- 각 component의 기여도 정량화
- SHAP 또는 feature importance

---

## 💡 논문 작성 팁

### Abstract/Introduction에 포함할 수치
- "Hierarchical architecture achieves **5.04%** relative improvement (p < 0.001)"
- "sentence-roberta encoder outperforms BERT by **4.03%** (p < 0.001)"
- "Results validated across **10 random seeds** (42-51)"

### Results Section
- Best configurations table (LaTeX 코드 제공됨)
- Statistical test results table (p-values, effect sizes)
- Ablation study results

### Discussion
- Why hierarchical works better (document-level context)
- Why sentence-roberta is best (pre-trained on sentence tasks)
- Limitations (computational cost, variance)

### Figures 추천
1. Bar chart: Flat vs Hierarchical comparison
2. Box plot: Seed variance visualization
3. Heatmap: Configuration performance matrix
4. Line plot: Performance vs. model size

---

## 📝 LaTeX Table 예시 (논문용)

```latex
\begin{table*}[t]
\centering
\caption{Performance Comparison: Flat vs. Hierarchical Architecture (n=10 seeds)}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Task} & \textbf{Architecture} & \textbf{Weighted F1} & \textbf{Macro F1} & \textbf{Accuracy} & \textbf{p-value} \\
\midrule
\multirow{2}{*}{4-way} 
  & Flat & $0.6517 \pm 0.0116$ & $0.6435 \pm 0.0129$ & $0.6495 \pm 0.0117$ & \multirow{2}{*}{$<0.001$***} \\
  & Hierarchical & $\mathbf{0.6846 \pm 0.0109}$ & $\mathbf{0.6783 \pm 0.0116}$ & $\mathbf{0.6836 \pm 0.0107}$ & \\
\midrule
\multirow{2}{*}{6-way} 
  & Flat & $0.5269 \pm 0.0104$ & $0.5147 \pm 0.0115$ & $0.5294 \pm 0.0108$ & \multirow{2}{*}{$<0.001$***} \\
  & Hierarchical & $\mathbf{0.5424 \pm 0.0119}$ & $\mathbf{0.5254 \pm 0.0113}$ & $\mathbf{0.5415 \pm 0.0123}$ & \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## ✅ 체크리스트

### 데이터 수집 완료
- [x] 1,520 experiments collected
- [x] All 10 seeds (42-51) present
- [x] No missing configurations

### 통계 분석 완료
- [x] Main hypothesis tested (Flat vs Hier)
- [x] Encoder comparison
- [x] Aggregator comparison
- [x] Classifier comparison
- [x] Layer selection analysis
- [x] Pooling strategy analysis
- [x] Effect sizes calculated
- [x] LaTeX tables generated

### 추가 작업 필요
- [ ] Multiple comparison correction
- [ ] Effect size confidence intervals
- [ ] Visualizations (plots)
- [ ] Confusion matrices
- [ ] Error analysis
- [ ] Computational cost analysis

---

## 🎉 결론

**Hierarchical document modeling이 flat baseline보다 통계적으로 유의하게 우수한 성능을 보임.**

핵심 수치:
- 4-way: **68.46%** (hierarchical) vs 65.17% (flat) → +5.04% 
- 6-way: **54.24%** (hierarchical) vs 52.69% (flat) → +2.95%
- p < 0.001, Cohen's d > 2.4 (large effect)

이는 문서 수준의 구조적 정보를 효과적으로 활용하는 hierarchical architecture의 우수성을 입증합니다.
