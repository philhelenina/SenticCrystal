# 📋 Changelog

All notable changes to SenticCrystal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-09-10 - **Major Refactoring Release**

### 🎯 **Major Achievements**
- **Config146 Performance**: Achieved 71.91% Macro-F1 (99.9% of 72% target)
- **Bayesian Integration**: Added uncertainty quantification with confidence scoring
- **Code Refactoring**: 75% duplicate code reduction through inheritance hierarchy
- **Saturn Cloud**: A100 optimization with 4-5x performance improvement

### ✨ **Added**
- **New Architecture**: `BaseEmbeddingGenerator` abstract class for code reuse
- **Config146Generator**: Refactored implementation with multi-K support
- **BayesianConfig146Generator**: True Bayesian inference with uncertainty quantification
- **Saturn Cloud Support**: A100 SXM4-80GB optimized environment
- **Dialogue Boundary Awareness**: Context windows respect conversation boundaries
- **Documentation Structure**: Organized docs/ directory with categorized documentation
- **Confidence Filtering**: Automatic quality control with human review flagging

### 🔄 **Changed**
- **Project Structure**: Reorganized into clear src/, docs/, backup/ hierarchy
- **Embedding Generation**: Efficient multi-K value generation (4-5x faster)
- **Context Windows**: Dynamic K values based on dialogue length vs fixed K=6
- **Configuration**: Unified Config146 optimal settings as defaults
- **Error Handling**: Centralized logging and exception management

### 🗑️ **Removed**
- **Duplicate Files**: Moved redundant embedding generators to backup/
- **WordNet Duplicates**: Consolidated 4 locations to 1 (scripts/wn-affect-1.0)
- **Unused Code**: Moved GCP data loader to backup (not currently used)

### 🐛 **Fixed**
- **-1 Label Handling**: Confirmed correct exclusion from training, inclusion in context
- **Memory Leaks**: Improved model sharing across generators
- **Import Dependencies**: Resolved circular import issues with refactoring

### 🔧 **Technical Details**

#### **Config146 Optimal Settings**
```python
{
    'apply_word_pe': False,
    'pooling_method': 'weighted_mean', 
    'apply_sentence_pe': False,
    'combination_method': 'sum',
    'bayesian_method': 'context_lstm'
}
```

#### **Performance Improvements**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Code Duplication | 60% | 15% | -75% |
| Multi-K Generation | K×time | 1×time | 4-5x faster |
| Memory Usage | 3×models | 1×shared | -66% |
| Saturn Cloud Training | 8 hours | 1.5 hours | 5x faster |

#### **New APIs**
```python
# Multi-K efficient generation
generator.generate_multiple_k_embeddings(texts, ids, k_values=[0,2,4,6])

# Bayesian with uncertainty
embeddings, uncertainty_info = bayesian_gen.generate_embeddings(
    texts, ids, return_uncertainty=True
)

# Confidence filtering
high_conf, low_conf, uncertainty = bayesian_gen.generate_with_confidence_filtering(
    texts, ids, confidence_threshold=0.8
)
```

### 📊 **Experimental Results**
- **IEMOCAP 4-way**: 71.91% Macro-F1, 72.04% Accuracy, 72.36% Weighted-F1
- **Focal Loss**: Recovery from 30.9% → 69.94% accuracy
- **Bayesian Uncertainty**: Automatic quality control with confidence scoring
- **Context Analysis**: K-turn strategies validated across multiple seeds

---

## [1.0.0] - 2025-09-09 - **Initial Production Release**

### ✨ **Added**
- **Core Architecture**: Sentence-RoBERTa + WordNet-Affect combination
- **IEMOCAP Support**: 4-way and 6-way emotion classification
- **Context Modeling**: K-turn context windows for conversational understanding
- **Bayesian Models**: Simple and advanced Bayesian neural networks
- **Focal Loss**: Class imbalance handling for emotion datasets
- **Configuration System**: 240+ experimental configurations
- **Preprocessing Pipeline**: Unified IEMOCAP and MELD data handling

### 🎯 **Initial Performance**
- **Baseline Accuracy**: ~66% on IEMOCAP 4-way
- **Context Impact**: K=6 achieving ~70% accuracy
- **Model Variants**: MLP, LSTM, and Bayesian implementations

### 🏗️ **Architecture**
- **Feature Extraction**: `src/features/sroberta_module.py`, `wnaffect_module.py`
- **Models**: `src/models/simple_bayesian.py`, `sequential_bayesian.py`
- **Preprocessing**: `src/utils/preprocessing.py`, `focal_loss.py`
- **Generation**: `src/data_preprocessing/embedding_generator.py`

---

## [0.1.0] - 2025-09-08 - **Research Prototype**

### ✨ **Added**
- **Proof of Concept**: Basic emotion recognition pipeline
- **Data Loading**: IEMOCAP CSV processing
- **Sentence Embeddings**: Basic Sentence-BERT integration
- **Classification**: Simple MLP for emotion prediction

### 🎯 **Research Goals**
- Establish baseline performance on IEMOCAP
- Explore context window strategies
- Investigate Bayesian approaches for uncertainty

---

## 🔮 **Upcoming Features (Roadmap)**

### **v2.1.0 - Information Theory Enhancement**
- [ ] **Mutual Information**: Optimal feature combination weights
- [ ] **Information Bottleneck**: Dynamic context window selection
- [ ] **Entropy-based**: Active learning sample selection

### **v2.2.0 - Multi-Modal Expansion** 
- [ ] **Audio Features**: Prosodic and acoustic features
- [ ] **Visual Features**: Facial expression integration
- [ ] **Cross-Modal**: Attention mechanisms

### **v3.0.0 - Production Deployment**
- [ ] **Real-time**: Streaming emotion recognition
- [ ] **API**: REST API for production integration
- [ ] **Monitoring**: Performance tracking and alerting

---

## 📝 **Migration Guide**

### **From v1.x to v2.0**
```python
# OLD API (deprecated but still works)
from src.data_preprocessing.embedding_generator import Config146EmbeddingGenerator

# NEW API (recommended)
from src.data_preprocessing.config146_generator import Config146EmbeddingGenerator

# New features
generator.generate_multiple_k_embeddings(...)  # Multi-K support
generator.generate_embeddings(..., dialogue_ids=...)  # Boundary awareness
```

### **File Structure Changes**
```bash
# Files moved to backup/
src/data_preprocessing/embedding_generator.py → backup/data_preprocessing/
src/data_preprocessing/bayesian_embedding_generator.py → backup/data_preprocessing/
src/utils/gcp_data_loader.py → backup/utils/

# Documentation reorganized
*.md → docs/{experiments,analysis,setup}/

# WordNet consolidated
scripts/wn-affect-1.1/ → backup/wordnet_data/
scripts/wn-domains-3.2/ → backup/wordnet_data/
data/wn-domains/ → backup/wordnet_data/
```

---

**For detailed technical information, see the [documentation](docs/) directory.**