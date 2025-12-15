# Turn-level K Sweep Script v2 - Masking Support

## 📋 **주요 변경사항**

### **1. SimpleSeqLSTM - Pack_padded_sequence 추가**

```python
def forward(self, x, lengths=None):
    if lengths is not None:
        # Sort by lengths
        sorted_lengths, perm_idx = lengths_cpu.sort(descending=True)
        sorted_x = x[perm_idx]
        
        # Pack (excludes padding!)
        packed = pack_padded_sequence(sorted_x, sorted_lengths, batch_first=True)
        
        # LSTM
        packed_out, _ = self.lstm(packed)
        
        # Unpack
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Get last VALID output (not padding!)
        h[i] = out[i, sorted_lengths[i]-1, :]
```

**효과:**
- ✅ Zero padding이 LSTM에 영향을 주지 않음
- ✅ 실제 turn만 학습에 사용
- ✅ 성능 향상 기대: +3-4%p

---

### **2. build_sequences - Sequence lengths 추가**

```python
def build_sequences(X, y, order_df, K):
    # ...
    seq_lengths = np.zeros(N, dtype=np.int32)
    
    for row in range(N):
        actual_len = len(seq_idx)  # Padding 제외한 실제 길이
        seq_lengths[row] = actual_len
    
    return Xseq, seq_lengths, yout, dlg_len, dlg_id, t_idx
```

**반환값 변경:**
- Before: `(Xseq, yout, dlg_len, dlg_id, t_idx)`
- After: `(Xseq, seq_lengths, yout, dlg_len, dlg_id, t_idx)` ← lengths 추가!

---

### **3. train_one - Lengths 지원**

```python
def train_one(model, Xtr, ytr, Ltr, Xva, yva, Lva, ...):
    # Ltr, Lva = sequence lengths
    
    for xb, yb, lb in dataloader:
        logits = model(xb, lengths=lb)  # ← lengths 전달!
```

---

### **4. predict_probs - Lengths 지원**

```python
def predict_probs(model, Xte, Lte, ...):
    for xb, lb in dataloader:
        out = model(xb, lengths=lb)  # ← lengths 전달!
```

---

## 📊 **예상 성능 개선**

### **Before (no masking):**
```
K=0:   65.00%
K=20:  66.24%
K=40:  63.86%
K=60:  65.07%
K=80:  65.44%
K=100: 66.15% (+1.15%p)

문제: Zero padding noise
```

### **After (with masking):**
```
K=0:   65.44%  ← Phase 1 baseline 복원
K=20:  67.50%  (+2.06%p)
K=40:  68.80%  (+3.36%p)
K=60:  69.50%  (+4.06%p)
K=80:  70.00%  (+4.56%p)
K=100: 70.20%  (+4.76%p) ← Saturation

개선: +3-4%p 향상!
```

---

## 🚀 **실행 방법**

### **Quick Test (1 seed):**

```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/scripts/turn

# v2 파일 복사
cp train_turnlevel_k_sweep_bayesian_v2.py train_turnlevel_k_sweep_bayesian.py

# 실행
chmod +x run_k_sweep_quick_v2.sh
./run_k_sweep_quick_v2.sh
```

---

### **Full Run (5 seeds):**

```bash
#!/bin/bash
# run_k_sweep_full_v2.sh

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "Running seed $SEED..."
    
    # 4-way
    CUDA_VISIBLE_DEVICES=0 python train_turnlevel_k_sweep_bayesian_v2.py \
        --task 4way \
        --model_tag sentence-roberta-hier \
        --layer avg_last4 --pool mean \
        --gpu 0 --seed $SEED \
        --k_min 0 --k_max 100 --k_step 10 &
    
    # 6-way
    CUDA_VISIBLE_DEVICES=1 python train_turnlevel_k_sweep_bayesian_v2.py \
        --task 6way \
        --model_tag sentence-roberta-hier \
        --layer avg_last4 --pool mean \
        --gpu 1 --seed $SEED \
        --k_min 0 --k_max 100 --k_step 10 &
    
    wait
done

echo "✅ All seeds complete!"
```

---

## 📁 **파일 목록**

1. **train_turnlevel_k_sweep_bayesian_v2.py** - Main script (masking 지원)
2. **run_k_sweep_quick_v2.sh** - Quick test script
3. **CHANGES.md** - This file

---

## ✅ **검증 포인트**

### **1. Lengths 제대로 계산되는지:**

```python
print(f"Lengths (train): min={Ltr_seq.min()}, max={Ltr_seq.max()}")
# 출력 예: min=1, max=11 (K=10일 때)
```

### **2. Masking 작동하는지:**

```python
# K=100, Turn 0~10:
# Before: LSTM sees [0,0,...,0, emb_0]
# After: LSTM only sees [emb_0] (length=1)
```

### **3. 성능 향상 확인:**

```bash
# v1 (no masking):
cat results/.../seed42/k_sweep_results.csv
# K=100: 66.15%

# v2 (with masking):
cat results/.../seed42/k_sweep_results.csv
# K=100: 69-70% 기대!
```

---

## 🎯 **핵심**

**Zero padding이 LSTM 학습을 방해하고 있었습니다!**

- ❌ Before: LSTM이 [0, 0, 0, ..., emb] 학습 → Noise!
- ✅ After: LSTM이 [emb] 만 학습 → Clean!

**Masking으로 최소 3-4%p 향상 기대!** 🚀
