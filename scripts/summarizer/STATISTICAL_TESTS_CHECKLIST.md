# 통계 테스트 체크리스트 및 가이드

## ✅ 완료된 통계 테스트

### 1. 주요 가설 검정 (Main Hypothesis)
✅ **TEST 1: Flat vs Hierarchical**
- **4-way**: p < 0.001, Cohen's d = 2.92 (large effect)
- **6-way**: p < 0.001, Cohen's d = 2.41 (large effect)
- **결론**: Hierarchical이 통계적으로 매우 유의하게 우수함

### 2. Encoder 비교
✅ **TEST 2: Encoder Comparison (Flat only)**
- sentence-roberta vs bert-base: p < 0.001, d = 1.87
- sentence-roberta vs roberta-base: p < 0.001, d = 0.87
- roberta-base vs bert-base: p < 0.001, d = 0.86
- **결론**: sentence-roberta > roberta-base > bert-base

### 3. Aggregator 비교
✅ **TEST 3: Aggregator Comparison (Hierarchical only)**
- mean, attn, sum, expdecay: 유의한 차이 없음
- lstm aggregator: 유의하게 낮은 성능 (p < 0.001)
- **결론**: mean/attn 권장, lstm 제외

### 4. Classifier 비교
✅ **TEST 4: Classifier Comparison**
- 4-way: MLP와 LSTM 간 유의한 차이 없음
- 6-way flat: MLP가 약간 우수 (p = 0.006)
- **결론**: 두 classifier 모두 사용 가능

### 5. Layer Selection
✅ **TEST 5: Layer Selection**
- 4-way hierarchical: avg_last4가 약간 우수 (p = 0.007)
- 6-way hierarchical: last가 약간 우수 (p = 0.001)
- **결론**: Task-dependent

### 6. Pooling Strategy
✅ **TEST 6: Pooling Strategy**
- mean pooling이 가장 안정적
- wmean_pos_rev도 비슷한 성능
- **결론**: mean 또는 wmean_pos_rev 권장

---

## 🔬 추가로 수행해야 할 통계 테스트

### 1. Multiple Comparison Correction ⚠️ **중요**

현재 많은 pairwise comparison을 수행했으므로 다중 비교 보정이 필요합니다.

#### 방법 1: Bonferroni Correction
```python
from scipy.stats import mannwhitneyu

# Original p-values
p_values = [0.000220, 0.000116, ...]  # 수행한 모든 테스트의 p-value

# Bonferroni correction
n_tests = len(p_values)
alpha = 0.05
bonferroni_threshold = alpha / n_tests

# Check significance
corrected_results = [p < bonferroni_threshold for p in p_values]
```

#### 방법 2: Holm-Bonferroni (더 권장)
```python
from statsmodels.stats.multitest import multipletests

p_values = [...]  # 모든 p-value 리스트
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
```

#### 방법 3: FDR (False Discovery Rate)
```python
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

**권장사항**: 
- Main hypothesis (Flat vs Hierarchical)는 p < 0.001이므로 어떤 보정을 적용해도 유의함
- Secondary analysis는 Holm-Bonferroni 적용 권장

---

### 2. Effect Size Confidence Intervals ⚠️ **권장**

Cohen's d의 신뢰구간을 계산하여 효과 크기의 불확실성을 정량화합니다.

```python
from scipy import stats
import numpy as np

def cohens_d_ci(group1, group2, confidence=0.95):
    """Calculate Cohen's d with confidence interval"""
    
    n1, n2 = len(group1), len(group2)
    
    # Cohen's d
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + 
                          (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    d = mean_diff / pooled_std
    
    # Standard error of d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Confidence interval
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d
    
    return d, (ci_lower, ci_upper)

# Example usage
hier_data = [...]  # Best hierarchical results
flat_data = [...]  # Best flat results

d, (ci_lower, ci_upper) = cohens_d_ci(hier_data, flat_data)
print(f"Cohen's d = {d:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")
```

---

### 3. Bootstrap Analysis ⚠️ **권장**

Bootstrap을 사용하여 평균 차이의 신뢰구간을 계산합니다.

```python
from scipy.stats import bootstrap

def bootstrap_mean_difference(group1, group2, n_bootstrap=10000):
    """Bootstrap confidence interval for mean difference"""
    
    def statistic(x, y):
        return np.mean(x) - np.mean(y)
    
    # Bootstrap
    rng = np.random.default_rng()
    res = bootstrap(
        (group1, group2),
        statistic,
        n_resamples=n_bootstrap,
        confidence_level=0.95,
        random_state=rng,
        method='percentile'
    )
    
    return res.confidence_interval

# Example
hier_data = df_raw[(df_raw['task']=='4way') & (df_raw['type']=='hierarchical') & 
                   ...]['weighted_f1'].values
flat_data = df_raw[(df_raw['task']=='4way') & (df_raw['type']=='flat') & 
                   ...]['weighted_f1'].values

ci = bootstrap_mean_difference(hier_data, flat_data)
print(f"Mean difference 95% CI: [{ci.low:.4f}, {ci.high:.4f}]")
```

---

### 4. Power Analysis ⚠️ **선택적**

현재 샘플 크기(n=10)가 충분한지 사후 검정력 분석을 수행합니다.

```python
from statsmodels.stats.power import ttest_power

def post_hoc_power(group1, group2, alpha=0.05):
    """Calculate post-hoc statistical power"""
    
    # Cohen's d
    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2)
    
    # Calculate power
    power = ttest_power(
        effect_size=abs(d),
        nobs=(n1 + n2) / 2,
        alpha=alpha,
        alternative='two-sided'
    )
    
    return power

# Example
power = post_hoc_power(hier_data, flat_data)
print(f"Statistical Power: {power:.4f}")
# If power > 0.8, sample size is adequate
```

---

### 5. Normality Test (선택적)

Mann-Whitney U test는 non-parametric이지만, 데이터가 정규분포를 따르는지 확인할 수 있습니다.

```python
from scipy.stats import shapiro

def test_normality(data, name="Data"):
    """Test for normality using Shapiro-Wilk test"""
    
    stat, p_value = shapiro(data)
    
    print(f"{name}:")
    print(f"  Shapiro-Wilk statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print(f"  ✅ Data appears normally distributed (p > 0.05)")
    else:
        print(f"  ⚠️  Data may not be normally distributed (p < 0.05)")
    
    return p_value

# Test all groups
test_normality(hier_4way_data, "4-way Hierarchical")
test_normality(flat_4way_data, "4-way Flat")
```

---

### 6. Homogeneity of Variance Test (선택적)

```python
from scipy.stats import levene

def test_homogeneity(group1, group2):
    """Test for homogeneity of variance using Levene's test"""
    
    stat, p_value = levene(group1, group2)
    
    print(f"Levene's Test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print(f"  ✅ Variances are homogeneous (p > 0.05)")
    else:
        print(f"  ⚠️  Variances are not homogeneous (p < 0.05)")
    
    return p_value

test_homogeneity(hier_data, flat_data)
```

---

### 7. Kruskal-Wallis Test (다중 그룹 비교)

세 개 이상의 그룹을 동시에 비교할 때 사용합니다.

```python
from scipy.stats import kruskal

def kruskal_wallis_test(*groups):
    """Kruskal-Wallis H-test for multiple groups"""
    
    stat, p_value = kruskal(*groups)
    
    print(f"Kruskal-Wallis H-test:")
    print(f"  H-statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"  *** At least one group is significantly different")
    else:
        print(f"  No significant difference among groups")
    
    return p_value

# Example: Compare all encoders
bert_data = df_raw[(df_raw['encoder']=='bert-base') & ...]['weighted_f1'].values
roberta_data = df_raw[(df_raw['encoder']=='roberta-base') & ...]['weighted_f1'].values
sroberta_data = df_raw[(df_raw['encoder']=='sentence-roberta') & ...]['weighted_f1'].values

kruskal_wallis_test(bert_data, roberta_data, sroberta_data)
```

---

### 8. Friedman Test (반복 측정)

동일한 configuration에서 여러 seed의 결과를 비교할 때 사용합니다.

```python
from scipy.stats import friedmanchisquare

def friedman_test(data_matrix):
    """
    Friedman test for repeated measures
    
    data_matrix: shape (n_configurations, n_seeds)
    """
    
    stat, p_value = friedmanchisquare(*data_matrix.T)
    
    print(f"Friedman Test:")
    print(f"  Chi-square statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    return p_value

# Example: Compare different seeds for best config
best_config_data = df_raw[
    (df_raw['encoder']=='sentence-roberta') & 
    (df_raw['layer']=='last') & 
    ...
].pivot(index='seed', columns='task', values='weighted_f1')

friedman_test(best_config_data.values)
```

---

## 📊 추가 분석 권장 사항

### 9. Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
```

### 10. Error Analysis by Class
```python
def analyze_per_class_performance(results_dict):
    """Analyze performance for each class"""
    
    for class_name, metrics in results_dict.items():
        print(f"\nClass: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")
```

### 11. Computational Cost Analysis
```python
def analyze_computational_cost():
    """Compare training time and memory usage"""
    
    # Load timing data
    flat_time = [...]
    hier_time = [...]
    
    print(f"Average Training Time:")
    print(f"  Flat: {np.mean(flat_time):.2f}s ± {np.std(flat_time):.2f}s")
    print(f"  Hierarchical: {np.mean(hier_time):.2f}s ± {np.std(hier_time):.2f}s")
    
    # Statistical test
    u, p = mannwhitneyu(hier_time, flat_time)
    print(f"  Mann-Whitney U test: p = {p:.4f}")
```

### 12. Learning Curve Analysis
```python
def plot_learning_curves(train_losses, val_losses):
    """Plot training and validation curves"""
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
```

---

## 🎯 우선순위별 권장 사항

### 🔴 **High Priority (필수)**
1. ✅ Multiple comparison correction (Bonferroni/Holm)
2. ✅ Effect size confidence intervals
3. ⚠️ Bootstrap analysis for robustness

### 🟡 **Medium Priority (권장)**
4. ⚠️ Power analysis (sample size adequacy)
5. ⚠️ Confusion matrix analysis
6. ⚠️ Per-class performance analysis

### 🟢 **Low Priority (선택)**
7. Normality tests (이미 non-parametric 사용)
8. Homogeneity of variance tests
9. Computational cost comparison
10. Learning curve analysis

---

## 📝 논문 작성 체크리스트

### Results Section
- [x] Report best configurations with mean ± std
- [x] Include p-values and effect sizes
- [x] Provide comparison tables
- [ ] Add confusion matrices
- [ ] Include per-class performance

### Statistical Reporting
- [x] Report test statistics (U, p-value)
- [x] Report effect sizes (Cohen's d)
- [ ] Report confidence intervals
- [ ] Apply multiple comparison correction
- [ ] Report statistical power

### Figures
- [x] Bar chart (Flat vs Hier)
- [x] Box plot (seed variance)
- [x] Encoder comparison
- [x] Aggregator comparison
- [x] Heatmap (top configurations)
- [ ] Confusion matrices
- [ ] Learning curves

### Tables
- [x] Best configurations (Top 5)
- [x] Statistical test results
- [ ] Ablation study results
- [ ] Computational cost comparison

---

## 💡 실행 스크립트 예시

```python
#!/usr/bin/env python3
"""
Complete statistical testing with corrections
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Load data
df_raw = pd.read_csv('all_results_combined.csv')

# Collect all p-values from pairwise tests
p_values = []
test_names = []

# Main hypothesis
for task in ['4way', '6way']:
    # Get best configs...
    u, p = stats.mannwhitneyu(hier_data, flat_data, alternative='greater')
    p_values.append(p)
    test_names.append(f"Hier vs Flat ({task})")

# Encoder comparisons
encoders = ['bert-base', 'roberta-base', 'sentence-roberta']
for i in range(len(encoders)):
    for j in range(i+1, len(encoders)):
        # Get data...
        u, p = stats.mannwhitneyu(data1, data2, alternative='greater')
        p_values.append(p)
        test_names.append(f"{encoders[j]} vs {encoders[i]}")

# Apply Holm-Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

# Print results
print("Test Results with Holm-Bonferroni Correction:")
print("-" * 80)
for name, p_orig, p_corr, sig in zip(test_names, p_values, p_corrected, reject):
    sig_str = "***" if sig else "n.s."
    print(f"{name:40s} p={p_orig:.6f} -> {p_corr:.6f} {sig_str}")
```

---

## ✅ 최종 체크리스트

### 데이터 분석
- [x] 1,520 experiments collected
- [x] Statistical tests performed
- [x] Effect sizes calculated
- [ ] Multiple comparison correction
- [ ] Confidence intervals

### 시각화
- [x] Main results plot
- [x] Comparison plots
- [x] Heatmaps
- [ ] Confusion matrices

### 논문 자료
- [x] LaTeX tables
- [x] Statistical test summary
- [ ] Supplementary materials

이제 이 가이드를 따라 추가 통계 분석을 수행하면 됩니다!
