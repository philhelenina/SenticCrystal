# 결과 수집 가이드 (Results Aggregation Guide)

## 📋 개요

`aggregate_all_results_n10.py` 스크립트는 n=10 실험 결과를 자동으로 수집하고 분석합니다.

### 수집 내용
- **Total**: 1,200 experiments
- **Flat**: 720 experiments (3 encoders × 2 layers × 3 pools × 2 classifiers × 2 tasks × 10 seeds)
- **Hierarchical**: 480 experiments (2 layers × 2 pools × 5 aggregators × 2 classifiers × 2 tasks × 10 seeds)

---

## 🚀 사용 방법

### Step 1: 경로 수정 (필수!)

스크립트 상단의 `BASE_DIR` 변수를 수정하세요:

```python
# 이 부분을 본인의 경로에 맞게 수정
BASE_DIR = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
```

### Step 2: 실행

```bash
# 스크립트 실행
python aggregate_all_results_n10.py
```

---

## 📂 입력 구조

스크립트는 다음 디렉토리 구조에서 결과를 읽어옵니다:

```
results_n10/
├── 4way/
│   ├── flat/
│   │   ├── bert-base/
│   │   │   ├── last/
│   │   │   │   ├── mean/
│   │   │   │   │   ├── mlp/
│   │   │   │   │   │   ├── seed_42/results.json
│   │   │   │   │   │   ├── seed_43/results.json
│   │   │   │   │   │   └── ... (through seed_51)
│   │   │   │   │   └── lstm/
│   │   │   │   ├── attn/
│   │   │   │   └── wmean_pos_rev/
│   │   │   └── avg_last4/
│   │   ├── roberta-base/
│   │   └── sentence-roberta/
│   └── hierarchical/
│       └── sentence-roberta-hier/
│           ├── last/
│           │   ├── mean/
│           │   │   ├── mean/  (aggregator)
│           │   │   │   ├── mlp/
│           │   │   │   └── lstm/
│           │   │   ├── attn/  (aggregator)
│           │   │   ├── sum/
│           │   │   ├── expdecay/
│           │   │   └── lstm/  (aggregator)
│           │   └── wmean_pos_rev/
│           └── avg_last4/
└── 6way/
    └── (same structure)
```

---

## 📊 출력 파일

### 전체 결과 (Combined Analysis)

`results_n10/analysis/` 디렉토리에 저장:

1. **all_results_combined.csv**
   - 모든 실험의 raw data (각 seed별 결과)
   - 1,200 rows (모든 실험)

2. **summary_combined.csv**
   - 각 설정별 평균 ± 표준편차
   - Configuration별로 집계

3. **flat_vs_hierarchical.csv**
   - Flat vs Hierarchical 비교
   - Task별, Type별 평균/최대값

4. **encoder_comparison.csv**
   - Encoder 비교 (Flat baseline만)
   - BERT vs RoBERTa vs Sentence-RoBERTa

5. **aggregator_comparison.csv**
   - Aggregator 비교 (Hierarchical만)
   - mean, attn, sum, expdecay, lstm

6. **classifier_comparison.csv**
   - Classifier 비교 (MLP vs LSTM)
   - Flat/Hierarchical 모두 포함

7. **seed_variance.csv**
   - Seed 간 분산 분석
   - Task/Type별 평균 std

### Task별 결과

`results_n10/4way/analysis/` 및 `results_n10/6way/analysis/`:

1. **all_results.csv** - Task별 raw data
2. **summary_statistics.csv** - Task별 summary
3. **best_flat.csv** - Top 10 flat configurations
4. **best_hierarchical.csv** - Top 10 hierarchical configurations

---

## 📈 출력 예시

실행 시 콘솔에 다음과 같은 정보가 출력됩니다:

```
================================================================================
AGGREGATING ALL RESULTS (n=10 seeds)
================================================================================
Results base: /path/to/results_n10
Expected: 1,200 total experiments
  - Flat: 720 (3 encoders × 2 layers × 3 pools × 2 classifiers × 2 tasks × 10 seeds)
  - Hierarchical: 480 (2 layers × 2 pools × 5 aggregators × 2 classifiers × 2 tasks × 10 seeds)
================================================================================

================================================================================
PROCESSING 4WAY
================================================================================

📁 Collecting results...
  Found 360 flat result files for 4way
  Found 240 hierarchical result files for 4way

✅ Total collected: 600 results
   Flat: 360
   Hierarchical: 240
   Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
   N unique seeds: 10

📊 Computing summary statistics...

🏆 Best Configurations:

Top 5 Flat Baseline (4WAY):
----------------------------------------------------------------------------------------------------
#1
  Encoder:     sentence-roberta
  Layer:       avg_last4
  Pool:        mean
  Classifier:  lstm
  Weighted F1: 0.8234 ± 0.0156
  Macro F1:    0.7891 ± 0.0178
  Accuracy:    0.8245 ± 0.0162
  N seeds:     10

...

🏆 OVERALL BEST CONFIGURATIONS:
================================================================================

4WAY - FLAT:
  Encoder:     sentence-roberta
  Layer:       avg_last4
  Pool:        mean
  Classifier:  lstm
  Weighted F1: 0.8234 ± 0.0156
  Macro F1:    0.7891 ± 0.0178
  Accuracy:    0.8245 ± 0.0162

4WAY - HIERARCHICAL:
  Encoder:     sentence-roberta-hier
  Layer:       avg_last4
  Pool:        wmean_pos_rev
  Aggregator:  mean
  Classifier:  mlp
  Weighted F1: 0.8456 ± 0.0134
  Macro F1:    0.8123 ± 0.0145
  Accuracy:    0.8467 ± 0.0139
```

---

## 🔍 검증 체크리스트

스크립트 실행 후 확인 사항:

### 1. 파일 개수 확인
```bash
# Flat 결과 확인 (720개 예상)
find results_n10/4way/flat -name "results.json" | wc -l
find results_n10/6way/flat -name "results.json" | wc -l

# Hierarchical 결과 확인 (480개 예상)
find results_n10/4way/hierarchical -name "results.json" | wc -l
find results_n10/6way/hierarchical -name "results.json" | wc -l
```

### 2. Seed 확인
각 configuration마다 정확히 10개의 seed (42-51)가 있어야 합니다:

```bash
# 예시: sentence-roberta/avg_last4/mean/mlp
ls results_n10/4way/flat/sentence-roberta/avg_last4/mean/mlp/
# 출력: seed_42/ seed_43/ ... seed_51/
```

### 3. 출력 파일 확인
```bash
# Analysis 디렉토리가 생성되었는지 확인
ls results_n10/analysis/
ls results_n10/4way/analysis/
ls results_n10/6way/analysis/
```

---

## 📊 결과 분석 예시

### Python으로 결과 읽기

```python
import pandas as pd

# 전체 결과 읽기
df_all = pd.read_csv("results_n10/analysis/all_results_combined.csv")
print(f"Total experiments: {len(df_all)}")

# Summary 읽기
df_summary = pd.read_csv("results_n10/analysis/summary_combined.csv")

# 4-way flat 최고 성능
best_4way_flat = df_summary[
    (df_summary['task'] == '4way') &
    (df_summary['type'] == 'flat')
].nlargest(1, 'weighted_f1_mean')

print(best_4way_flat)
```

### 통계 분석

```python
from scipy import stats

# Flat vs Hierarchical 비교 (4-way)
flat_results = df_all[
    (df_all['task'] == '4way') &
    (df_all['type'] == 'flat') &
    (df_all['encoder'] == 'sentence-roberta') &
    (df_all['layer'] == 'avg_last4') &
    (df_all['pool'] == 'mean') &
    (df_all['classifier'] == 'lstm')
]['weighted_f1'].values

hier_results = df_all[
    (df_all['task'] == '4way') &
    (df_all['type'] == 'hierarchical') &
    (df_all['encoder'] == 'sentence-roberta-hier') &
    (df_all['layer'] == 'avg_last4') &
    (df_all['pool'] == 'wmean_pos_rev') &
    (df_all['aggregator'] == 'mean') &
    (df_all['classifier'] == 'mlp')
]['weighted_f1'].values

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(hier_results, flat_results, alternative='greater')
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Hierarchical is significantly better (p < 0.05)")
else:
    print("⚠️  No significant difference")
```

---

## ⚠️ 문제 해결

### "Directory not found" 에러
```bash
# results_n10 디렉토리가 있는지 확인
ls -la results_n10/

# 경로 수정 필요
# 스크립트의 BASE_DIR 변수를 올바른 경로로 수정
```

### "No results found" 경고
```bash
# 실험이 완료되었는지 확인
find results_n10/ -name "results.json" | wc -l

# 로그 확인
tail -100 n10_gpu0_flat.log
```

### Seed 개수가 10개가 아닌 경우
```bash
# 특정 config의 seed 확인
ls results_n10/4way/flat/sentence-roberta/last/mean/mlp/

# 누락된 seed 찾기
for seed in {42..51}; do
  if [ ! -d "results_n10/4way/flat/sentence-roberta/last/mean/mlp/seed_${seed}" ]; then
    echo "Missing: seed_${seed}"
  fi
done
```

---

## 💡 팁

### 1. 부분 실행
특정 task만 분석하고 싶다면:

```python
# 스크립트 수정
for task in ["4way"]:  # 6way 제외
    ...
```

### 2. 추가 분석
스크립트는 기본 분석만 제공합니다. 더 상세한 분석을 위해:

```python
# Layer 비교
layer_comp = df_summary.groupby(['task', 'type', 'layer']).agg({
    'weighted_f1_mean': 'mean'
}).round(4)

# Pool 비교
pool_comp = df_summary.groupby(['task', 'type', 'pool']).agg({
    'weighted_f1_mean': 'mean'
}).round(4)
```

### 3. 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Seed별 분산 시각화
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_all[df_all['task']=='4way'], 
            x='type', y='weighted_f1', hue='classifier')
plt.title('4-way Classification: Flat vs Hierarchical')
plt.savefig('comparison.png')
```

---

## ✅ 완료 기준

다음 조건이 모두 만족되면 결과 수집이 완료된 것입니다:

- ✅ 1,200개의 results.json 파일 존재
- ✅ 각 configuration마다 정확히 10개 seed (42-51)
- ✅ 모든 분석 CSV 파일 생성
- ✅ Best configurations 출력 확인
- ✅ P-value < 0.05 (통계적 유의성)

---

## 📦 백업

결과 수집 후 백업 권장:

```bash
# 결과 압축
tar -czf results_n10_backup_$(date +%Y%m%d).tar.gz results_n10/

# Analysis 파일만 백업
tar -czf analysis_backup_$(date +%Y%m%d).tar.gz results_n10/*/analysis/

# 확인
tar -tzf results_n10_backup_*.tar.gz | head
```

---

## 🆘 도움이 필요하면

문제가 발생하면:
1. 로그 파일 확인 (`*.log`)
2. 디렉토리 구조 확인
3. Seed 개수 확인
4. 경로 설정 재확인

Good luck! 🍀
