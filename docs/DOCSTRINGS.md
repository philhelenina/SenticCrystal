# Docstrings for Code Files

Copy-paste these docstrings to the top of each Python script.

---

## generate_sroberta_npz_4way.py

```python
"""
Generate Sentence-RoBERTa embeddings for IEMOCAP 4-way classification (flat).

This script produces utterance-level embeddings [N, 768] for the 4-way emotion
classification task (anger, happiness, sadness, neutral).

LAYER COMBINATION OPTIONS:
    - last: Uses only the final transformer layer. This captures the most
            task-specific semantic representations but may lose some lexical info.
    - avg_last4: Averages layers 9-12. Balances syntactic (lower) and semantic
                 (upper) information for richer representations.
    - last4_scalar_*: Weighted combination with presets:
        * up [1,2,3,4]: Emphasizes deeper layers (more contextual)
        * down [4,3,2,1]: Emphasizes shallower layers (more lexical)
        * top2 [0,1,1,0]: Focuses on layers 10-11 (often optimal in probing)

TOKEN POOLING OPTIONS:
    - cls: Standard [CLS] token pooling; sentence-level representation
    - mean: Masked average; robust, treats all tokens equally
    - attn: Attention-weighted; learns token importance dynamically
    - wmean_pos: Front-weighted; hypothesis that opening words set emotional tone
    - wmean_pos_rev: End-weighted; hypothesis that conclusions carry emotion climax
    - wmean_exp_*: Exponential decay (fast/med/slow τ values); models recency bias
    - wmean_idf: IDF-weighted; emphasizes rare, informative words

Output: NPZ files at {out_root}/{layer}/{pool}/{split}.npz
"""
```

---

## generate_sroberta_npz_6way.py

```python
"""
Generate Sentence-RoBERTa embeddings for IEMOCAP 6-way classification (flat).

Identical to 4-way version but includes all 6 emotion classes:
anger, happiness, sadness, neutral, excited, frustrated.

The 6-way task tests the model's ability to distinguish fine-grained emotions:
    - excited vs. happy: Both positive valence, but different arousal levels
    - frustrated vs. angry: Both negative, but frustrated implies helplessness

See generate_sroberta_npz_4way.py for detailed option descriptions.
"""
```

---

## generate_sroberta_hier_npz.py

```python
"""
Generate hierarchical Sentence-RoBERTa embeddings (auto-detect 4way/6way).

HIERARCHICAL APPROACH RATIONALE:
    Unlike flat embeddings that treat an utterance as a single text block,
    hierarchical embeddings first encode each sentence separately, then
    aggregate them. This preserves sentence-level granularity which is
    important because:

    1. Emotions can shift within an utterance ("I was happy but then...")
    2. Different sentences may carry different emotional weight
    3. Sentence order/position may indicate emotional progression

OUTPUT SHAPE: [N, S_max, 768] where S_max is max sentences per utterance
    - Also saves 'lengths' array for masking during training

SENTENCE SPLITTING:
    Conservative regex-based splitting on .!? followed by whitespace.
    Preserves utterance integrity while enabling sentence-level analysis.

This script auto-detects 4way/6way from the data_dir path.
"""
```

---

## generate_sroberta_hier_npz_4way.py / 6way.py

```python
"""
Generate hierarchical Sentence-RoBERTa embeddings for IEMOCAP {4/6}-way.

Task-specific version of generate_sroberta_hier_npz.py.
Hardcodes paths for {4/6}-way data to avoid detection errors.

See generate_sroberta_hier_npz.py for hierarchical approach rationale.
"""
```

---

## train_npz_classifier_4way_verbose.py

```python
"""
Train flat classifier for IEMOCAP 4-way emotion recognition.

This script trains MLP or LSTM classifiers on pre-computed Sentence-RoBERTa
embeddings. "Flat" means the input is utterance-level [N, D] without
sentence-level hierarchy.

CLASSIFIER OPTIONS:
    - MLP: 2-layer feedforward (Linear-ReLU-Dropout-Linear)
           Fast training, interpretable, works well when embeddings are
           already semantically rich from the transformer.

    - LSTM: Single-layer LSTM treating embedding dims as sequence
            Can capture patterns across embedding dimensions, though
            the semantic ordering assumption is weak for transformers.

TRAINING DETAILS:
    - CrossEntropyLoss with inverse-frequency class weights (handles imbalance)
    - Adam optimizer with optional weight decay
    - Early stopping on validation loss (patience=60 default)
    - 10 seeds (42-51) for statistical significance testing

VERBOSE MODE:
    Prints real-time training progress including:
    - Per-epoch train/val loss and accuracy
    - Per-class F1 scores on test set
    - Confusion matrix visualization
"""
```

---

## train_npz_classifier_6way_verbose.py

```python
"""
Train flat classifier for IEMOCAP 6-way emotion recognition.

6-way classification is more challenging than 4-way because:
    1. More classes to distinguish (6 vs 4)
    2. Similar emotions must be separated (excited/happy, frustrated/angry)
    3. Class imbalance is more severe (neutral often dominates)

See train_npz_classifier_4way_verbose.py for detailed training description.
"""
```

---

## train_npz_hier_classifier_4way.py / 6way.py

```python
"""
Train hierarchical classifier for IEMOCAP {4/6}-way emotion recognition.

HIERARCHICAL MODEL ARCHITECTURE:
    Input [B, S, D] → Aggregator → [B, D'] → Classifier → [B, num_classes]

AGGREGATOR OPTIONS (sentence → utterance):
    - mean: Simple average. Baseline that treats all sentences equally.
            Works when emotion is distributed across the utterance.

    - sum: Summation. Preserves magnitude; longer utterances get stronger
           signal. Useful when emotion accumulates across sentences.

    - expdecay: Exponential decay weighting (λ=0.5, reverse=True by default).
                Emphasizes later sentences based on the hypothesis that
                speakers conclude with their final emotional state.

    - attn: Learned attention weights. A linear layer learns which sentences
            are most emotionally salient. Adds parameters but is adaptive.

    - lstm: LSTM aggregation. Processes sentences sequentially, capturing
            how emotion evolves through the utterance. Returns final hidden
            state as the utterance representation.

CLASSIFIER OPTIONS:
    - MLP: Standard feedforward; fast and effective
    - LSTM: Additional sequential modeling after aggregation

The combination of aggregator + classifier creates a two-stage architecture
that separately models sentence-level and utterance-level patterns.
"""
```

---

## train_npz_hier_fused_classifier.py

```python
"""
Train classifier on SenticNet-fused hierarchical embeddings.

FUSION APPROACH:
    Combines Sentence-RoBERTa embeddings with lexicon-based features from
    SenticNet/WordNet-Affect. This tests whether explicit sentiment lexicons
    improve neural embedding representations.

FUSION MODES:
    - concat: Simple concatenation [sroberta; senticnet]
              Preserves all information but increases dimensionality

    - proj128: Project both to 128d then concatenate
               Reduces dimensions while maintaining both signals

    - zeropad768: Pad SenticNet features to 768d
                  Enables direct comparison with unfused models

LOSS OPTIONS:
    - ce: Standard CrossEntropyLoss
    - focal: Focal Loss (γ=2.0) for handling class imbalance
             Down-weights easy examples, focuses on hard ones

This script is task-agnostic (4way/6way specified via --task flag).
"""
```

---

## Shell Scripts (run_*.sh)

```bash
###############################################################################
# GPU Experiment Runner
#
# PURPOSE:
#   Parallelizes experiments across multiple GPUs to maximize throughput.
#   Each script handles a specific subset of the experimental grid.
#
# EXPERIMENTAL GRID:
#   - Encoders: bert-base, roberta-base, sentence-roberta (flat only)
#   - Layers: avg_last4, last
#   - Pools: mean, attn, wmean_pos_rev
#   - Aggregators: mean, sum, expdecay, attn, lstm (hier only)
#   - Classifiers: mlp, lstm
#   - Seeds: 42-51 (n=10 for statistical significance)
#
# GPU DISTRIBUTION RATIONALE:
#   Flat experiments (720 runs) split by task × classifier:
#     GPU 0: 4way MLP (180 runs)
#     GPU 1: 4way LSTM (180 runs)
#     GPU 2: 6way MLP (180 runs)
#     GPU 3: 6way LSTM (180 runs)
#
#   Hierarchical experiments (480 runs) split by task × aggregator subset:
#     GPU 0: 4way (mean, lstm, attn-mlp)
#     GPU 1: 4way (attn-lstm, expdecay, sum)
#     GPU 2: 6way (mean, lstm, attn-mlp)
#     GPU 3: 6way (attn-lstm, expdecay, sum)
#
# OUTPUT:
#   - Per-run results in results_n10/{task}/{flat|hierarchical}/...
#   - Summary logs in results_n10/{task}/summary/
###############################################################################
```
