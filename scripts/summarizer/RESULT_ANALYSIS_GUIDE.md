# Baseline Results Analysis - Quick Start 🔍

**목적:** 실행된 baseline 실험 결과를 집계하고 분석

---

## 📋 사전 확인

### 결과 파일 확인:
```bash
# 4way 결과 개수 확인
find results/baseline/4way -name "results.json" | wc -l
# Expected: 120 (4 encoders × 2 layers × 3 pools × 5 seeds)

# 6way 결과 개수 확인
find results/baseline/6way -name "results.json" | wc -l
# Expected: 120

# 전체
find results/baseline -name "results.json" | wc -l
# Expected: 240 (MLP만) 또는 480 (MLP + LSTM)
```

---

## 🚀 실행 방법

### **Option 1: 자동 실행 (추천)**

```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/scripts

# 분석 스크립트 복사
cp /mnt/user-data/outputs/aggregate_baseline_results.py .
cp /mnt/user-data/outputs/visualize_baseline_results.py .
cp /mnt/user-data/outputs/analyze_baseline_results.sh .
chmod +x analyze_baseline_results.sh

# 실행 (1-2분 소요)
./analyze_baseline_results.sh
```

### **Option 2: 단계별 실행**

```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/scripts

# Step 1: 결과 집계
python aggregate_baseline_results.py

# Step 2: 시각화
python visualize_baseline_results.py
```

---

## 📊 생성되는 파일들

### **CSV Files:**
```
results/baseline/
├── all_results_combined.csv          # 모든 결과 (raw, seed별)
├── summary_combined.csv               # 요약 통계 (mean ± std)
├── encoder_comparison.csv             # Encoder별 평균
├── model_comparison.csv               # MLP vs LSTM
├── 4way/
│   ├── all_results.csv
│   ├── summary_statistics.csv
│   ├── summary_table.csv              # 포맷된 표
│   └── best_configs.csv               # Top 10 configurations
└── 6way/
    └── (동일 구조)
```

### **Figures:**
```
results/baseline/figures/
├── encoder_comparison.png             # Encoder 비교
├── heatmap_4way.png                   # 4way 성능 히트맵
├── heatmap_6way.png                   # 6way 성능 히트맵
├── top_configs_4way.png               # 4way Top 10
├── top_configs_6way.png               # 6way Top 10
└── layer_pool_analysis.png            # Layer/Pool 분석
```

---

## 📈 결과 확인 방법

### **1. 터미널 출력 확인:**

스크립트 실행 시 다음 정보가 출력됩니다:
- Top 10 configurations (weighted F1 기준)
- Overall best configuration
- Encoder comparison
- Model comparison (MLP vs LSTM)

### **2. CSV 파일 확인:**

```bash
# Best configs 확인 (weighted F1 순)
head -n 11 results/baseline/4way/best_configs.csv | column -t -s,

# Encoder 비교
cat results/baseline/encoder_comparison.csv | column -t -s,

# MLP vs LSTM 비교
cat results/baseline/model_comparison.csv | column -t -s,
```

### **3. Python으로 상세 분석:**

```python
import pandas as pd

# Load summary
df = pd.read_csv('results/baseline/summary_combined.csv')

# Best configuration by weighted_f1
best_4way = df[df['task'] == '4way'].sort_values('weighted_f1_mean', ascending=False).iloc[0]
print("Best 4way config:")
print(f"  Encoder: {best_4way['encoder']}")
print(f"  Layer: {best_4way['layer']}")
print(f"  Pool: {best_4way['pool']}")
print(f"  Model: {best_4way['model']}")
print(f"  WF1: {best_4way['weighted_f1_mean']:.4f} ± {best_4way['weighted_f1_std']:.4f}")

# Compare encoders
encoder_avg = df.groupby(['task', 'encoder'])['weighted_f1_mean'].mean().unstack()
print("\nEncoder comparison:")
print(encoder_avg)
```

---

## 🎯 주요 분석 포인트

### **1. Best Encoder 선정**
- Weighted F1 기준 상위 encoder 확인
- Sentence-RoBERTa가 일반적으로 우수
- Task별 차이 확인 (4way vs 6way)

### **2. Layer Aggregation**
- avg_last4 vs last
- 어느 것이 더 안정적인가?

### **3. Pooling Strategy**
- mean vs attn vs wmean_pos_rev
- Position-based pooling의 효과

### **4. Model Type**
- MLP vs LSTM
- Task complexity에 따른 차이

### **5. Variance Analysis**
- Seed간 표준편차 확인
- 안정성이 높은 configuration 선택

---

## 📝 예상 결과

### **Typical Rankings:**

**4way (Expected):**
```
1. sentence-roberta / avg_last4 / wmean_pos_rev / lstm : 0.7450 ± 0.0045
2. sentence-roberta / avg_last4 / attn / lstm          : 0.7425 ± 0.0052
3. sentence-roberta / last / wmean_pos_rev / lstm      : 0.7398 ± 0.0048
...
```

**6way (Expected):**
```
1. sentence-roberta / avg_last4 / wmean_pos_rev / lstm : 0.7125 ± 0.0056
2. sentence-roberta / avg_last4 / attn / lstm          : 0.7098 ± 0.0063
3. roberta-base / avg_last4 / wmean_pos_rev / lstm     : 0.7045 ± 0.0059
...
```

---

## 🏆 다음 단계

### **Best Configuration 선정 후:**

1. **Stage 2 (Hierarchical Baseline) 실행**
   - Best encoder 사용
   - Best layer/pool 사용
   
2. **Stage 3 (Turn-level K-Sweep)**
   - Best flat configuration을 baseline으로
   - Context window 효과 분석

---

## ⚠️ 문제 해결

### **결과 파일이 없는 경우:**
```bash
# 실험이 완료되었는지 확인
ls -lh results/baseline/4way/sentence-roberta/avg_last4/mean/mlp/seed_42/

# 특정 configuration 재실행
python train_npz_classifier_4way.py \
  --layer avg_last4 \
  --pool mean \
  --model mlp \
  --emb_root "data/embeddings/4way/sentence-roberta" \
  --out_dir "results/baseline/4way/sentence-roberta/avg_last4/mean/mlp/seed_42" \
  --seed 42
```

### **일부 configuration만 있는 경우:**
- summary 통계에서 `n_seeds` 컬럼 확인
- 5개 미만이면 실험 미완료

### **Memory 오류:**
```python
# aggregate_baseline_results.py에서 청크 단위로 처리하도록 수정
# 또는 task별로 따로 실행
```

---

## 📚 참고

**Weighted F1 vs Macro F1:**
- Weighted F1: 클래스 불균형 고려 (더 중요)
- Macro F1: 모든 클래스 동등 (소수 클래스 성능 확인)

**표준편차 해석:**
- std < 0.005: 매우 안정적
- std < 0.01: 안정적
- std > 0.02: 불안정 (configuration 재검토)

**Encoder 선택 기준:**
1. Weighted F1 mean (1순위)
2. 표준편차 (안정성)
3. Computational cost (속도/메모리)

---

**분석 완료 후 결과를 확인하고 best configuration을 선정하세요!** 📊

이 configuration을 Stage 2 (Hierarchical)와 Stage 3 (Turn-level)에서 사용합니다.
