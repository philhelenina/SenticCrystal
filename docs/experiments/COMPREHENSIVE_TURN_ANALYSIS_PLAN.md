# SenticCrystal 종합 Turn Analysis 실험 계획

## 🎯 **실험 목표**

Config146 설정 + Forward-only Turn Context Window 전략을 통한 감정 인식 성능 최적화

## 📊 **Phase 1: Turn Analysis (현재 단계)**

### **1.1 임베딩 생성**
**Input:**
- IEMOCAP 4-way 데이터 (train/val/test)
- Config146 설정:
  ```json
  {
    "apply_word_pe": false,
    "pooling_method": "weighted_mean", 
    "apply_sentence_pe": false,
    "combination_method": "sum",
    "bayesian_method": "context_lstm"
  }
  ```
- WordNet-Affect (300차원) + Sentence-RoBERTa (768차원)
- Forward-only context: 현재 + 이전 K-1개 턴만 사용

**Output:**
- Config146 임베딩 파일들:
  ```
  embeddings/config146_proper/
  ├── 0turn/  # K=0 (현재 utterance만)
  ├── 2turn/  # K=2 (이전 1개 + 현재)
  ├── 4turn/  # K=4 (이전 3개 + 현재) 
  └── 6turn/  # K=6 (이전 5개 + 현재)
  ```

### **1.2 Context Window 전략 구현**

**전략 1: Cumulative Quantile**
- 각 대화별 temporal position 기반
- Q1 (0-25%): K ≤ 8
- Q2 (25-50%): K ≤ 15  
- Q3 (50-75%): K ≤ 25
- Q4 (75-100%): K ≤ 35

**전략 2: Pure Cumulative**
- K = utterance_position (dialogue 내 위치)
- 처음부터 현재까지 모든 이전 context 사용

**전략 3: Conservative Cumulative**  
- Pure cumulative의 보수적 버전
- K = min(utterance_position, max_K_limit)

**전략 4: Fixed Baselines**
- K=0, K=2, K=4, K=6 고정값

### **1.3 분류 모델 Architecture**

**기본 MLP 분류기:**
```python
MLPClassifier(
    input_size=768,          # S-RoBERTa + WNA (sum 결합)
    hidden_size=256,         # 첫 번째 hidden layer
    hidden_size2=128,        # 두 번째 hidden layer  
    num_classes=4,           # 4-way classification
    dropout=0.3,             # Regularization
    activation='ReLU'        # Activation function
)

# Training Configuration
optimizer = Adam(lr=0.001)   # Learning rate
loss = CrossEntropyLoss()    # Loss function
batch_size = 32              # Batch size
max_epochs = 200             # Maximum epochs
early_stopping = 10          # Patience
```

### **1.4 평가 메트릭**

**모든 전략에 대해 다음 메트릭 수집:**
1. **Accuracy** - 전체 정확도
2. **Macro F1-Score** - 클래스별 F1의 평균 
3. **Weighted F1-Score** - 클래스 빈도로 가중된 F1
4. **Per-class F1** - Angry, Happy, Sad, Neutral 각각
5. **Confusion Matrix** - 4x4 혼동 행렬
6. **Learning Curves** - Train/Val loss, accuracy 곡선

### **1.5 통계적 검증**
- **Multiple Random Seeds** (3-5회 실행)
- **Statistical Significance Testing** (t-test)
- **Error Analysis** - 클래스별 성능 분석

## 📈 **Expected Results**

**Target Performance Table:**
| Strategy | Accuracy | Macro-F1 | Weighted-F1 | 기대 성능 |
|----------|----------|----------|-------------|---------|
| Fixed K=0 | ~0.66 | ~0.66 | ~0.66 | Baseline |
| Fixed K=2 | ~0.67 | ~0.67 | ~0.67 | Baseline |
| Fixed K=4 | ~0.70 | ~0.70 | ~0.70 | Baseline |
| Fixed K=6 | ~0.72 | ~0.72 | ~0.72 | Current Best |
| **Cumulative Quantile** | **~0.75** | **~0.75** | **~0.76** | **Target** |
| **Pure Cumulative** | **~0.75** | **~0.75** | **~0.76** | **Target** |
| **Conservative Cumulative** | **~0.74** | **~0.74** | **~0.75** | **Target** |

## 📊 **Output 결과물**

### **1. 성능 분석 파일들**
```
results/turn_analysis_20250910/
├── comprehensive_results.json          # 모든 메트릭 결과
├── statistical_analysis.json           # 통계적 유의성 분석
├── per_class_performance.json          # 클래스별 성능 분석
└── confusion_matrices.json             # 모든 전략의 혼동행렬
```

### **2. 시각화 자료들**
```
figures/turn_analysis/
├── performance_comparison.png          # 전략별 성능 비교 차트
├── learning_curves_all_strategies.png  # 모든 전략의 학습 곡선
├── confusion_matrices_grid.png         # 혼동행렬 grid 시각화
├── statistical_significance.png        # 통계적 유의성 시각화
└── per_class_f1_breakdown.png         # 클래스별 F1 분해 분석
```

### **3. 종합 분석 보고서**
```
TURN_ANALYSIS_COMPREHENSIVE_RESULTS.md
├── Executive Summary
├── Methodology Details  
├── Performance Results
├── Statistical Validation
├── Error Analysis
├── Implementation Specifications
└── Next Steps for Optimization
```

## 🚀 **Phase 2: Optimization (이후 단계)**

Phase 1 완료 후 진행할 최적화 전략:

### **2.1 Bayesian Hyperparameter Optimization**
- **Target Parameters:**
  - Learning rate scheduling
  - Architecture depth/width
  - Dropout rates
  - Batch size optimization

### **2.2 Loss Function Optimization** 
- **Focal Loss** with gamma parameter tuning
- **Label Smoothing** for confidence calibration  
- **Weighted CrossEntropy** for class balancing

### **2.3 Architecture Comparison**
- **MLP vs LSTM** classifier 비교
- **Attention mechanisms** 추가
- **Residual connections** 적용

### **2.4 Performance Probing**
- **High-performance region** 탐색
- **Ensemble methods** 적용
- **Model fusion** strategies

## ⏱️ **실행 계획**

1. **Step 1** (2-3시간): Config146 임베딩 생성 (모든 K값)
2. **Step 2** (3-4시간): Cumulative context 전략 구현 및 실험
3. **Step 3** (1-2시간): 종합 결과 분석 및 시각화
4. **Step 4** (1시간): 보고서 및 문서화

**Total Estimated Time: 7-10시간**

## 🎯 **Success Criteria**

✅ **Phase 1 성공 기준:**
- [ ] 모든 turn 전략 실험 완료
- [ ] 통계적 유의성 확인 (+3% 이상 성능 향상)
- [ ] 모든 메트릭 및 시각화 자료 생성
- [ ] 재현 가능한 구현 및 문서화

✅ **Ready for Phase 2:**
- [ ] Baseline 대비 성능 향상 확인
- [ ] 최적 turn 전략 식별
- [ ] Optimization 방향성 결정