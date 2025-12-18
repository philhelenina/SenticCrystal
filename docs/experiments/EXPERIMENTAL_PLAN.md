# SenticCrystal: Complete Experimental Plan

## 🎯 **Current Status & Goals**

### **Current Achievement:**
- **Cumulative Context Window**: 74.89% Macro-F1 (simulated)
- **Improvement**: +4.14% over baseline K=6
- **Problem**: Need actual model training with all three metrics

### **Missing Requirements:**
- ❌ **Weighted-F1** scores 
- ❌ **Accuracy** scores
- ❌ **Actual model training** (not simulation)
- ❌ **Statistical validation**

## 📊 **Phase 1: Complete Metric Evaluation**

### **Experiment Setup**
```
├── Baseline Experiments
│   ├── Fixed K=2 (CrossEntropy, MLP)
│   ├── Fixed K=4 (CrossEntropy, MLP) 
│   ├── Fixed K=6 (CrossEntropy, MLP) ← Current best known
│   └── Fixed K=0 (No context, MLP)
│
├── Cumulative Context Experiments  
│   ├── Pure Cumulative (K=position)
│   ├── Conservative Cumulative (K=min(position, 15))
│   ├── Quantile-based Cumulative
│   └── Adaptive Threshold Cumulative
│
└── Evaluation Metrics (ALL experiments)
    ├── Accuracy
    ├── Macro-F1 
    ├── Weighted-F1
    ├── Per-class F1 (Angry, Happy, Sad, Neutral)
    └── Confusion Matrix
```

### **Data Configuration**
- **Dataset**: IEMOCAP 4-way (Angry, Happy, Sad, Neutral)
- **Architecture**: WordNet-Affect + Sentence-RoBERTa + MLP
- **Embeddings**: Config146 (proven effective)
- **Data Splits**: Standard train/val/test with -1 label filtering

### **Model Architecture (Standardized)**
```python
MLPClassifier(
    input_size=778,          # WNA(10) + S-RoBERTa(768)
    hidden_size=256,         # First hidden layer
    hidden_size2=128,        # Second hidden layer  
    num_classes=4,           # 4-way classification
    dropout=0.3,             # Regularization
    activation='ReLU'        # Activation function
)

# Training Configuration
optimizer = Adam(lr=0.001)   # Learning rate
loss = CrossEntropyLoss()    # Loss function (no Focal yet)
batch_size = 32              # Batch size
max_epochs = 200             # Maximum epochs
early_stopping = 10          # Patience
```

## 🔬 **Phase 2: Context Window Implementation**

### **Strategy 1: Fixed Context Baselines**
```python
def load_fixed_context_data(k_turns):
    # Load K-turn embeddings and labels
    # Use existing embeddings/config146/iemocap_4way/{k}turn/
    return train_data, val_data, test_data

experiments = ['0turn', '2turn', '4turn', '6turn']
```

### **Strategy 2: Cumulative Context Simulation**  
```python
def create_cumulative_context_data():
    # Method 1: Weighted combination of multiple K-turn data
    # Higher K gets more weight to simulate cumulative effect
    
    data_combination = {
        '2turn': weight=0.33,  # Early dialogue context
        '4turn': weight=0.67,  # Mid dialogue context  
        '6turn': weight=1.00   # Full context available
    }
    
    # Method 2: Per-utterance K assignment
    for utterance in dialogue:
        k_value = min(utterance_position, max_k)
        context_features = get_context_features(utterance, k_value)
    
    return cumulative_data
```

### **Strategy 3: Adaptive Context Windows**
```python
def adaptive_context_strategy(dialogue_length, utterance_position):
    # Quantile-based approach
    temporal_quantile = utterance_position / dialogue_length
    
    if temporal_quantile <= 0.25:      # Q1: Early
        recommended_k = min(8, utterance_position)
    elif temporal_quantile <= 0.50:    # Q2: Early-mid  
        recommended_k = min(15, utterance_position)
    elif temporal_quantile <= 0.75:    # Q3: Mid-late
        recommended_k = min(25, utterance_position)
    else:                              # Q4: Late
        recommended_k = min(35, utterance_position)
    
    return recommended_k
```

## 📈 **Phase 3: Comprehensive Evaluation**

### **Expected Results Table**
| Strategy | Accuracy | Macro-F1 | Weighted-F1 | Improvement | Status |
|----------|----------|----------|-------------|-------------|---------|
| **Fixed K=0** | ~0.66 | ~0.66 | ~0.66 | -3.5% | Baseline |
| **Fixed K=2** | ~0.67 | ~0.67 | ~0.67 | -2.5% | Baseline |  
| **Fixed K=4** | ~0.70 | ~0.70 | ~0.70 | -1.0% | Baseline |
| **Fixed K=6** | ~0.72 | ~0.72 | ~0.72 | 0.0% | **Current** |
| **Pure Cumulative** | **~0.75** | **~0.75** | **~0.76** | **+4.2%** | **Target** |
| **Quantile Cumulative** | **~0.74** | **~0.74** | **~0.75** | **+3.8%** | **Target** |
| **Conservative Cumulative** | **~0.74** | **~0.74** | **~0.75** | **+3.5%** | **Target** |

### **Validation Requirements**
- ✅ **Actual Model Training**: Real PyTorch training loops
- ✅ **Cross-validation**: Multiple random seeds (3-5 runs)
- ✅ **Statistical Testing**: t-test for significance  
- ✅ **Error Analysis**: Per-class performance breakdown
- ✅ **Computational Cost**: Training time comparison

## 🚀 **Phase 4: Optimization Pipeline**

### **Current Architecture Limitations**
```python
# PRE-OPTIMIZATION BASELINE (Current 74.89% target)
{
    'loss_function': 'CrossEntropyLoss',     # No class balancing
    'optimizer': 'Adam(lr=0.001)',          # Default learning rate
    'architecture': '2-layer MLP',          # Simple architecture
    'regularization': 'Dropout(0.3)',       # Basic regularization
    'batch_size': 32,                       # Default batch size
    'status': 'PRE-OPTIMIZATION'
}
```

### **Post-Baseline Optimization Plan** 
1. **Loss Function Optimization** (+1-2%)
   - Focal Loss for class imbalance
   - Label Smoothing for confidence calibration
   - Weighted CrossEntropy for direct balancing

2. **Bayesian Hyperparameter Optimization** (+1-2%)
   - Learning rate scheduling
   - Architecture depth/width tuning
   - Dropout rate optimization
   - Batch size optimization

3. **Advanced Architecture** (+0.5-1%)
   - Attention mechanisms
   - Residual connections
   - Batch normalization

**Expected Final Performance**: **76-78% Macro-F1**

## 💻 **Implementation Files**

### **File Structure**
```
/Users/helenjeong/Projects/SenticCrystal/
├── run_complete_context_experiments.py    # Main experiment runner
├── src/models/cumulative_classifier.py    # Cumulative context model
├── src/utils/context_strategies.py        # Context window strategies  
├── results/complete_results_{timestamp}.json
├── figures/complete_performance_analysis.png
└── COMPLETE_RESULTS.md                    # Updated results summary
```

### **Experiment Execution Plan**
```bash
# Step 1: Run complete baseline experiments
python run_complete_context_experiments.py --phase baseline

# Step 2: Run cumulative context experiments  
python run_complete_context_experiments.py --phase cumulative

# Step 3: Generate comprehensive analysis
python create_complete_analysis.py

# Step 4: Update presentation materials
python update_presentation_slides.py
```

## 📊 **Success Criteria**

### **Phase 1 Success** ✅
- [ ] All experiments run with actual model training
- [ ] All three metrics (Accuracy, Macro-F1, Weighted-F1) reported
- [ ] Statistical significance validated
- [ ] Per-class performance analyzed

### **Phase 2 Success** 🎯  
- [ ] Cumulative context achieves **74%+ Macro-F1**
- [ ] **+3%+ improvement** over K=6 baseline confirmed
- [ ] Weighted-F1 shows **similar or better** improvement
- [ ] Results reproducible across multiple runs

### **Phase 3 Success** 🚀
- [ ] Complete presentation materials updated
- [ ] Publication-ready figures generated
- [ ] Methodology thoroughly documented
- [ ] Ready for optimization phase

## ⚡ **Next Immediate Actions**

1. **Execute Phase 1**: Run `run_complete_context_experiments.py`
2. **Validate Results**: Confirm all metrics match expectations  
3. **Update Materials**: Regenerate all figures, tables, MD files
4. **Proceed to Optimization**: Begin Bayesian hyperparameter tuning

---

**Status**: 📋 Plan completed, ready for execution
**Priority**: 🔥 **HIGH** - Foundation for entire publication
**Timeline**: 2-3 hours for complete Phase 1-3 execution