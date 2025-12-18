# SenticCrystal Turn Context Optimization Report
**Date**: September 11, 2025  
**Platform**: Saturn Cloud A100 (2x 80GB)  
**Project**: Emotion Recognition with Context Windows  

## 🎯 Project Overview

**Goal**: Optimize context window size (K-turns) for emotion recognition in conversational data using Config146 architecture.

**Architecture**: 
- **Config146**: WordNet-Affect (300-dim) + Sentence-RoBERTa (768-dim) 
- **Combination**: Sum method → 768-dim final embeddings
- **Models**: LSTM (sequence) + MLP (context window)
- **Dataset**: IEMOCAP 4-way classification (angry, happy, neutral, sad)

---

## 📊 Experimental Pipeline

### Phase 1: Baseline Establishment ✅
**Local Testing (Mac M4)**:
- K=0 (Baseline): 45.85% accuracy (MLP only)
- K=2: 66.32% accuracy (LSTM), 65.67% accuracy (MLP)
- **Key Finding**: +20%p improvement with context

### Phase 2: Bayesian Hyperparameter Optimization ✅
**Saturn Cloud A100 Results**:

**Optimization Details**:
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 50 Bayesian optimization trials
- **Objective**: Multi-objective (40% accuracy + 30% macro-F1 + 30% weighted-F1)
- **Search Space**:
  ```python
  {
      'learning_rate': (1e-5, 1e-2),        # log scale
      'batch_size': [32, 64, 128, 256, 512], # categorical  
      'hidden_size': (128, 512),             # step=64
      'dropout_rate': (0.1, 0.8),            # float
      'weight_decay': (1e-6, 1e-1),         # log scale
      'num_epochs': (50, 200),               # int
      'early_stopping_patience': (5, 20)     # int
  }
  ```

**🏆 Optimal Hyperparameters Found**:
```python
{
    'learning_rate': 0.00021556012144898146,
    'batch_size': 64,
    'hidden_size': 192,
    'dropout_rate': 0.7129363737801503,
    'weight_decay': 1.7083020624786403e-05,
    'num_epochs': 69,
    'early_stopping_patience': 20
}
```

**Performance with Optimal Parameters**:
- **K=2 MLP Performance**: Further validation needed
- **Optimization Time**: ~1-2 hours on A100
- **WandB Integration**: Full experiment tracking implemented

### Phase 3: Comprehensive Turn Analysis 🔄
**Planned Experiments**:

**Baseline Analysis** (K=0,2,4,6,8,10):
- Individual K-value performance
- Saturation point detection
- Model comparison (LSTM vs MLP)

**Cumulative Context Strategies**:
1. **Pure Cumulative**: Linear weight increase with K
   - Weights: {2: 0.2, 4: 0.4, 6: 0.6, 8: 0.8, 10: 1.0}
2. **Conservative Cumulative**: Peak at K=6, then decline  
   - Weights: {2: 0.3, 4: 0.7, 6: 1.0, 8: 0.8, 10: 0.6}
3. **Quantile Cumulative**: Balanced representation
   - Weights: {2: 0.25, 4: 0.5, 6: 0.75, 8: 1.0, 10: 0.9}
4. **Optimal Cumulative**: Performance-based optimization
   - Weights: {2: 0.1, 4: 0.3, 6: 0.8, 8: 0.6, 10: 0.4}

**Current Status**: ⚠️ **Debugging Phase** - Script integration issues

---

## 🛠️ Technical Implementation

### Core Scripts Developed:

1. **`bayesian_hyperparameter_optimization.py`** ✅
   - Optuna + WandB integration
   - Multi-objective optimization
   - Hyperparameter importance analysis

2. **`train_turn_classifier.py`** ✅ 
   - Forward-only context windows
   - K=2,4,6 support
   - Optimal hyperparameter integration

3. **`train_baseline_classifier.py`** ✅
   - K=0 baseline experiments  
   - Pure embedding classification
   - Comprehensive evaluation metrics

4. **`comprehensive_experiments_with_wandb.py`** 🔄
   - Automated experiment orchestration
   - Saturation point analysis
   - Cumulative strategy testing

### Architecture Updates:

**Enhanced Classifiers** (`classifiers.py`):
```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, num_classes=4, 
                 num_layers=1, dropout_rate=0.5):
        # LSTM with optimized dropout integration

class ContextMLP(nn.Module):  
    def __init__(self, context_size, embedding_dim=768, hidden_size=256,
                 num_classes=4, dropout_rate=0.5):
        # MLP for flattened context windows
```

### Data Processing:

**Forward-Only Context Windows**:
```python
def create_forward_only_context_windows(embeddings, df, k_value):
    # Use [t-k, t-k+1, ..., t-1, t] (no future information)
    start_idx = max(0, i - k_value)
    end_idx = i + 1
    context_embeddings = file_embeddings[start_idx:end_idx]
```

**Label Handling**:
- Training/Testing: Exclude -1 labels
- Context: Include -1 labels for realistic context
- 4-way mapping: {ang, hap, neu, sad}

---

## 📈 Performance Tracking

### WandB Integration:
- **Real-time monitoring**: Training curves, validation metrics
- **Hyperparameter tracking**: Parameter importance analysis  
- **Experiment comparison**: Automated result aggregation
- **Artifact management**: Model checkpoints, confusion matrices

### Evaluation Metrics:
- **Primary**: Accuracy, Macro-F1, Weighted-F1
- **Per-class**: Precision, Recall, F1-score for each emotion
- **Confusion Matrix**: Detailed error analysis
- **Training Dynamics**: Loss curves, convergence analysis

---

## 🚧 Current Issues & Debugging

### Identified Problems:

1. **Script Integration**: 
   - K-value constraints in argument parsing
   - Missing parameter propagation
   - Path resolution issues

2. **Experiment Orchestration**:
   - Baseline script compatibility  
   - Error handling for failed experiments
   - Result file path inconsistencies

### Next Steps:

1. **Fix Script Compatibility**:
   - Extend K-value support to [0,2,4,6,8,10]
   - Ensure all hyperparameters propagate correctly
   - Standardize result file naming

2. **Resume Comprehensive Analysis**:
   - Complete baseline experiments (K=0,2,4,6,8,10)
   - Run cumulative context strategies
   - Perform saturation point analysis

3. **Advanced Optimization** (Future):
   - Focal Loss vs CrossEntropyLoss comparison
   - Architecture search (layer depth, attention mechanisms)
   - Multi-modal fusion optimization

---

## 📁 File Structure

```
saturn_cloud_deployment/
├── bayesian_hyperparameter_optimization.py   # ✅ Bayesian optimization
├── train_turn_classifier.py                  # ✅ Turn experiments  
├── train_baseline_classifier.py             # ✅ Baseline experiments
├── comprehensive_experiments_with_wandb.py   # 🔄 Main orchestrator
├── classifiers.py                           # ✅ Model architectures
├── embeddings_saturn.py                     # ✅ Config146 embedding generation
├── environment.yml                          # ✅ Dependencies (optuna, wandb)
└── results/                                 # Experiment outputs
    ├── bayesian_optimization/               # Hyperparameter search results
    ├── turn_experiments/                    # K-value experiment results
    └── baseline_classifiers/                # K=0 baseline results
```

---

## 🎯 Research Questions

1. **Saturation Point**: At what K-value does performance plateau?
2. **Model Comparison**: LSTM vs MLP effectiveness with different context sizes
3. **Cumulative Strategies**: Which context combination strategy works best?
4. **Computational Efficiency**: Performance vs computational cost trade-offs
5. **Generalization**: Do optimal K-values transfer to other emotion datasets?

---

## 📊 Expected Outcomes

**Baseline Performance** (Local Results):
- K=0: ~46% accuracy
- K=2: ~66% accuracy (+20%p)
- K=4,6: TBD (Saturn Cloud with optimal hyperparameters)

**Optimization Impact**:
- Bayesian optimization found significantly different hyperparameters
- High dropout rate (0.71) suggests overfitting prevention important
- Lower learning rate (0.0002) for stable convergence
- Batch size 64 optimal for A100 memory utilization

**Research Contributions**:
1. Systematic K-value optimization for conversational emotion recognition
2. Bayesian hyperparameter optimization methodology for emotion AI
3. Comprehensive comparison of context aggregation strategies
4. Practical guidelines for context window sizing in dialogue systems

---

*Report generated on Saturn Cloud A100 platform with comprehensive WandB tracking.*