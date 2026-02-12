# 📊 결과 집계 및 시각화 가이드

## 🎯 개요

실험이 완료되면 다음 두 스크립트를 사용하여 결과를 집계하고 시각화합니다:

1. **aggregate_all_results.py** - Flat + Hierarchical 결과 집계
2. **visualize_all_results.py** - 그래프 및 시각화 생성

---

## 📦 파일 준비

### 1. 스크립트 복사
```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/scripts

# 집계 및 시각화 스크립트 복사
cp /path/to/downloads/aggregate_all_results.py .
cp /path/to/downloads/visualize_all_results.py .
```

### 2. 실험 결과 디렉토리 구조

실험이 완료되면 다음과 같은 구조가 생성됩니다:

```
results/
├── baseline/                    # Flat baseline
│   ├── 4way/
│   │   ├── bert-base/
│   │   │   └── avg_last4/
│   │   │       └── mean/
│   │   │           ├── mlp/
│   │   │           │   ├── seed_42/
│   │   │           │   │   └── results.json
│   │   │           │   └── seed_43/
│   │   │           └── lstm/
│   │   ├── roberta-base/
│   │   └── sentence-roberta/
│   └── 6way/
│       └── (동일 구조)
└── hier_baseline/               # Hierarchical
    ├── 4way/
    │   └── sentence-roberta-hier/
    │       └── avg_last4/
    │           └── mean/
    │               ├── mean/        # aggregator
    │               │   ├── mlp/     # classifier
    │               │   │   ├── seed_42/
    │               │   │   │   └── results.json
    │               │   │   └── seed_43/
    │               │   └── lstm/
    │               ├── sum/
    │               ├── expdecay/
    │               ├── attn/
    │               └── lstm/
    └── 6way/
        └── (동일 구조)
```

---

## 🚀 사용 방법

### Step 1: 결과 집계

```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/scripts

# 모든 결과 집계 (Flat + Hierarchical)
python aggregate_all_results.py
```

**출력 예시:**
```
====================================================================================================
COMPREHENSIVE RESULTS AGGREGATION (Flat + Hierarchical)
====================================================================================================

====================================================================================================
Processing 4WAY
====================================================================================================

📦 Collecting Flat Baseline...
  Found 180 flat result files for 4way

📦 Collecting Hierarchical...
  Found 300 hierarchical result files for 4way

✅ Total collected: 480 results
   Flat: 180
   Hierarchical: 300
   Seeds: 5

📊 Computing summary statistics...

🏆 Best Configurations:

📊 Top 5 Flat Baseline (4WAY):
====================================================================================================
 1. sentence-roberta     | avg_last4  | wmean_pos_rev   | lstm | WF1: 0.6455 ± 0.0023
 2. sentence-roberta     | avg_last4  | attn            | lstm | WF1: 0.6421 ± 0.0018
 3. sentence-roberta     | last       | wmean_pos_rev   | mlp  | WF1: 0.6398 ± 0.0025
 ...

📊 Top 5 Hierarchical (4WAY):
====================================================================================================
 1. sentence-roberta-hier | avg_last4  | mean     | lstm | mlp  | WF1: 0.6512 ± 0.0019
 2. sentence-roberta-hier | avg_last4  | attn     | lstm | mlp  | WF1: 0.6489 ± 0.0021
 ...
```

**생성되는 파일:**

```
results/analysis/
├── all_results_combined.csv          # 모든 raw 결과 (seed별)
├── summary_combined.csv               # 통계 요약 (mean ± std)
├── flat_vs_hierarchical.csv          # Flat vs Hierarchical 비교
├── encoder_comparison.csv             # Encoder 비교 (flat만)
├── aggregator_comparison.csv          # Aggregator 비교 (hier만)
├── classifier_comparison.csv          # MLP vs LSTM 비교
├── 4way/
│   ├── all_results.csv               # 4way 전체 결과
│   ├── summary_statistics.csv        # 4way 요약
│   ├── best_flat.csv                 # 4way flat top 10
│   └── best_hierarchical.csv         # 4way hier top 10
└── 6way/
    ├── all_results.csv
    ├── summary_statistics.csv
    ├── best_flat.csv
    └── best_hierarchical.csv
```

### Step 2: 시각화 생성

```bash
# 집계 완료 후 시각화 실행
python visualize_all_results.py
```

**출력 예시:**
```
====================================================================================================
COMPREHENSIVE RESULTS VISUALIZATION (Flat + Hierarchical)
====================================================================================================

Loaded 360 configurations
  Flat:         180
  Hierarchical: 180

Generating visualizations...

📊 Creating comparison plots...
✅ Saved: results/analysis/figures/flat_vs_hierarchical.png
✅ Saved: results/analysis/figures/encoder_comparison_flat.png
✅ Saved: results/analysis/figures/aggregator_comparison_hierarchical.png
✅ Saved: results/analysis/figures/classifier_comparison.png

📊 Creating top configurations plots...
✅ Saved: results/analysis/figures/top_configs_4way_flat.png
✅ Saved: results/analysis/figures/top_configs_4way_hierarchical.png
✅ Saved: results/analysis/figures/top_configs_6way_flat.png
✅ Saved: results/analysis/figures/top_configs_6way_hierarchical.png

📊 Creating heatmaps...
✅ Saved: results/analysis/figures/heatmap_flat_4way.png
✅ Saved: results/analysis/figures/heatmap_flat_6way.png
✅ Saved: results/analysis/figures/heatmap_hierarchical_4way.png
✅ Saved: results/analysis/figures/heatmap_hierarchical_6way.png
```

**생성되는 그래프:**

```
results/analysis/figures/
├── flat_vs_hierarchical.png           # Flat vs Hierarchical 비교
├── encoder_comparison_flat.png        # Encoder 비교 (flat)
├── aggregator_comparison_hierarchical.png  # Aggregator 비교 (hier)
├── classifier_comparison.png          # MLP vs LSTM 비교
├── top_configs_4way_flat.png          # 4way flat top 10
├── top_configs_4way_hierarchical.png  # 4way hier top 10
├── top_configs_6way_flat.png          # 6way flat top 10
├── top_configs_6way_hierarchical.png  # 6way hier top 10
├── heatmap_flat_4way.png              # 4way flat heatmap
├── heatmap_flat_6way.png              # 6way flat heatmap
├── heatmap_hierarchical_4way.png      # 4way hier heatmap
└── heatmap_hierarchical_6way.png      # 6way hier heatmap
```

---

## 📊 생성되는 시각화

### 1. Flat vs Hierarchical 비교
![flat_vs_hierarchical](example_flat_vs_hierarchical.png)
- 두 접근법의 평균 성능 비교
- 최고 성능(별표)과 평균 성능(막대) 표시

### 2. Encoder 비교 (Flat Baseline)
![encoder_comparison](example_encoder.png)
- bert-base, roberta-base, sentence-roberta 비교
- Weighted F1 및 Macro F1

### 3. Aggregator 비교 (Hierarchical)
![aggregator_comparison](example_aggregator.png)
- mean, sum, expdecay, attn, lstm 비교
- 문장 aggregation 방법의 효과

### 4. Classifier 비교
![classifier_comparison](example_classifier.png)
- MLP vs LSTM 비교
- Task 및 Type별로 분리

### 5. Top Configurations
![top_configs](example_top_configs.png)
- 상위 10개 설정 및 성능
- Error bar 포함

### 6. Heatmaps
![heatmap](example_heatmap.png)
- 모든 설정 조합의 성능 매트릭스
- 최적 조합 식별 용이

---

## 📈 결과 분석 팁

### 1. Best Configuration 찾기

```python
import pandas as pd

# Summary 파일 로드
df = pd.read_csv('results/analysis/summary_combined.csv')

# 4way flat baseline 최고 성능
best_4way_flat = df[
    (df['task'] == '4way') & 
    (df['type'] == 'flat')
].sort_values('weighted_f1_mean', ascending=False).head(1)

print(best_4way_flat[['encoder', 'layer', 'pool', 'classifier', 'weighted_f1_mean', 'weighted_f1_std']])
```

### 2. Encoder 효과 분석

```python
# Encoder별 평균 성능
encoder_avg = df[df['type'] == 'flat'].groupby('encoder')['weighted_f1_mean'].mean()
print(encoder_avg.sort_values(ascending=False))
```

### 3. Aggregator 효과 분석

```python
# Aggregator별 평균 성능
agg_avg = df[df['type'] == 'hierarchical'].groupby('aggregator')['weighted_f1_mean'].mean()
print(agg_avg.sort_values(ascending=False))
```

### 4. Statistical Significance Test

```python
from scipy import stats

# Flat vs Hierarchical t-test
flat_results = df[df['type'] == 'flat']['weighted_f1_mean']
hier_results = df[df['type'] == 'hierarchical']['weighted_f1_mean']

t_stat, p_value = stats.ttest_ind(flat_results, hier_results)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

---

## 🔍 트러블슈팅

### 문제 1: "No results found"
```bash
# 결과 파일 위치 확인
find results/ -name "results.json" | wc -l

# 예상: 480+ (flat 180 + hier 300)
```

### 문제 2: "KeyError: 'metrics'"
→ results.json 파일 형식 확인:
```bash
cat results/baseline/4way/bert-base/avg_last4/mean/mlp/seed_42/results.json
```

예상 형식:
```json
{
  "metrics": {
    "accuracy": 0.6234,
    "macro_f1": 0.6012,
    "weighted_f1": 0.6156
  },
  ...
}
```

### 문제 3: 일부 결과만 집계됨
→ 경로 구조 확인:
```bash
# Flat 구조
results/baseline/[task]/[encoder]/[layer]/[pool]/[model]/seed_[X]/results.json

# Hierarchical 구조
results/hier_baseline/[task]/[encoder]/[layer]/[pool]/[aggregator]/[classifier]/seed_[X]/results.json
```

---

## 📝 논문 작성 팁

### 1. Results Section에 포함할 표

```markdown
## Table 1: Best Configurations

| Task | Type         | Configuration                          | Weighted F1    |
|------|--------------|----------------------------------------|----------------|
| 4way | Flat         | sentence-roberta/avg_last4/wmean/lstm | 0.6455 ± 0.0023|
| 4way | Hierarchical | sentence-roberta-hier/.../lstm/mlp    | 0.6512 ± 0.0019|
| 6way | Flat         | ...                                    | ...            |
| 6way | Hierarchical | ...                                    | ...            |
```

### 2. Figures에 포함할 그래프

- Figure 1: Flat vs Hierarchical comparison
- Figure 2: Encoder comparison (flat)
- Figure 3: Aggregator comparison (hierarchical)
- Figure 4: Top 10 configurations

### 3. 통계적 유의성 언급

```markdown
Hierarchical models showed significantly better performance than flat baselines
(p < 0.05, paired t-test), with an average improvement of X% in weighted F1.
```

---

## ✅ 체크리스트

실험 완료 후:
- [ ] 모든 results.json 파일이 생성되었는가?
- [ ] aggregate_all_results.py 실행 완료?
- [ ] visualize_all_results.py 실행 완료?
- [ ] 생성된 CSV 파일 확인?
- [ ] 생성된 그래프 확인?
- [ ] Best configuration 식별?
- [ ] 결과를 백업했는가?

---

## 💾 결과 백업

```bash
# 전체 결과 압축
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/

# 분석 결과만 백업
tar -czf analysis_backup_$(date +%Y%m%d).tar.gz results/analysis/
```

---

완료! 🎉

질문이 있으면 언제든지 물어보세요!