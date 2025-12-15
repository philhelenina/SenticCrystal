# 🚀 SenticNet Experiments - Complete Guide (4× A100 GPUs)

## 📦 Generated Scripts

1. ✅ `make_sentic_fusion_hierarchical_npz.py` - Fusion creation
2. ✅ `run_senticnet_parallel.py` - Parallel launcher (4 GPUs)
3. ✅ `train_npz_hier_classifier.py` - Individual training
4. ✅ `analyze_senticnet_results.py` - Results analysis

---

## 🎯 Full Experiment (RoBERTa included!)

### Configuration:
```
Encoders: 3 (bert-base-hier, roberta-base-hier, sentence-roberta-hier)
Fusions:  2 (baseline, concat)
Tasks:    2 (4way, 6way)
Seeds:    10 (42-51)

Total: 3 × 2 × 2 × 10 = 120 runs
GPUs: 4 (30 runs each)
Time: ~2-3 hours per GPU
```

---

## 🚀 Step-by-Step Execution

### Step 1: Create Fusions (~10 minutes)

```bash
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment

# BERT + SenticNet
python make_sentic_fusion_hierarchical_npz.py --task 4way --encoder bert-base-hier --fusion_mode concat
python make_sentic_fusion_hierarchical_npz.py --task 6way --encoder bert-base-hier --fusion_mode concat

# RoBERTa + SenticNet
python make_sentic_fusion_hierarchical_npz.py --task 4way --encoder roberta-base-hier --fusion_mode concat
python make_sentic_fusion_hierarchical_npz.py --task 6way --encoder roberta-base-hier --fusion_mode concat

# Sentence-RoBERTa + SenticNet
python make_sentic_fusion_hierarchical_npz.py --task 4way --encoder sentence-roberta-hier --fusion_mode concat
python make_sentic_fusion_hierarchical_npz.py --task 6way --encoder sentence-roberta-hier --fusion_mode concat
```

---

### Step 2: Verify Setup (CRITICAL!)

```python
import numpy as np
from pathlib import Path

def verify_all():
    encoders = ['bert-base-hier', 'roberta-base-hier', 'sentence-roberta-hier']
    tasks = ['4way', '6way']
    
    print("Verifying fusion embeddings...")
    for encoder in encoders:
        for task in tasks:
            # Baseline
            baseline_path = f'data/embeddings/{task}/{encoder}/avg_last4/mean/train.npz'
            # Concat
            concat_path = f'data/embeddings/{task}/{encoder}-sentic-concat/avg_last4/mean/train.npz'
            
            if Path(baseline_path).exists():
                baseline_data = np.load(baseline_path)
                print(f"✓ {encoder:30s} {task} baseline: {baseline_data['embeddings'].shape}")
            else:
                print(f"✗ {encoder:30s} {task} baseline: NOT FOUND")
            
            if Path(concat_path).exists():
                concat_data = np.load(concat_path)
                expected_dim = 772
                actual_dim = concat_data['embeddings'].shape[2]
                status = '✓' if actual_dim == expected_dim else '✗'
                print(f"{status} {encoder:30s} {task} concat:   {concat_data['embeddings'].shape} (expected D=772)")
            else:
                print(f"✗ {encoder:30s} {task} concat:   NOT FOUND")
            print()

verify_all()
```

**Expected Output:**
```
✓ bert-base-hier                 4way baseline: (2953, 15, 768)
✓ bert-base-hier                 4way concat:   (2953, 15, 772) (expected D=772)

✓ bert-base-hier                 6way baseline: (4490, 15, 768)
✓ bert-base-hier                 6way concat:   (4490, 15, 772) (expected D=772)

... (all 12 configurations)
```

---

### Step 3: Test Run (Single Experiment)

```bash
# Test on GPU 0
python train_npz_hier_classifier.py \
    --task 4way \
    --encoder bert-base-hier \
    --seed 42 \
    --gpu 0 \
    --epochs 5

# Should complete in ~1 minute
# Check output for errors
```

---

### Step 4: Dry Run (Verify Configuration)

```bash
python run_senticnet_parallel.py --dry_run
```

**Expected Output:**
```
================================================================================
SENTICNET PARALLEL EXPERIMENTS
================================================================================
Total experiments: 120
  Encoders: ['bert-base-hier', 'roberta-base-hier', 'sentence-roberta-hier']
  Tasks: ['4way', '6way']
  Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
  Fusions: baseline, concat

Job distribution:
  GPU 0: 30 jobs
    Breakdown: {'bert-base-hier': 20, 'roberta-base-hier': 10}
  GPU 1: 30 jobs
    Breakdown: {'roberta-base-hier': 10, 'sentence-roberta-hier': 20}
  GPU 2: 30 jobs
    Breakdown: {'sentence-roberta-hier': 20, 'bert-base-hier': 10}
  GPU 3: 30 jobs
    Breakdown: {'bert-base-hier': 10, 'roberta-base-hier': 10, 'sentence-roberta-hier': 10}
================================================================================
```

---

### Step 5: Launch Full Experiment! 🚀

```bash
# Launch parallel training on 4 GPUs
nohup python run_senticnet_parallel.py > senticnet_experiments.log 2>&1 &

# Monitor progress
tail -f senticnet_experiments.log

# Monitor GPUs
watch -n 1 nvidia-smi
```

**Expected Runtime:** 2-3 hours

---

### Step 6: Analyze Results

```bash
# After all experiments complete
python analyze_senticnet_results.py
```

**Outputs:**
```
results/senticnet_analysis/
├── improvements_by_encoder.png          # Bar plots
├── complementarity_analysis.png         # Scatter plot
├── improvements_summary.csv             # Statistics
├── all_results.csv                      # Raw data
└── ANALYSIS_REPORT.md                   # Full report
```

---

## 📊 Real-time Monitoring

### Check Progress:
```bash
# Count completed experiments
ls results/senticnet_experiments/*.json | wc -l

# Expected: 120 files when complete
```

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

**Expected:**
- All 4 GPUs should be at 80-100% utilization
- Memory usage: ~2-4 GB per GPU

### Check Results:
```bash
# View results as they complete
tail -20 results/senticnet_experiments/results.csv
```

---

## ⚠️ Troubleshooting

### Problem: GPU out of memory
**Solution:**
```bash
# Reduce batch size in train_npz_hier_classifier.py
--batch_size 16  # Default is 32
```

### Problem: Process killed
**Solution:**
```bash
# Check system memory
free -h

# Run on fewer GPUs
python run_senticnet_parallel.py --n_gpus 2
```

### Problem: Fusion file not found
**Solution:**
```bash
# Re-run fusion creation for that encoder
python make_sentic_fusion_hierarchical_npz.py \
    --task 4way \
    --encoder bert-base-hier \
    --fusion_mode concat
```

### Problem: Training script not found
**Solution:**
```bash
# Make sure you're in the correct directory
cd /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment

# Copy scripts to correct location
cp train_npz_hier_classifier.py /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/
```

---

## 🎯 Success Criteria

### ✅ Experiment Success:
- [ ] All 120 experiments complete
- [ ] results.csv has 120 rows
- [ ] No errors in log file
- [ ] Analysis script runs successfully

### ✅ Complementarity Confirmed:
- [ ] BERT improvement > Sentence-RoBERTa improvement
- [ ] Negative correlation < -0.5
- [ ] p < 0.05 for improvements
- [ ] Effect size > 1%

### Example Success Scenario:
```
bert-base-hier:
  baseline: 0.621
  concat:   0.645
  improvement: +0.024 (+3.9%) ✓

roberta-base-hier:
  baseline: 0.645  
  concat:   0.658
  improvement: +0.013 (+2.0%) ✓

sentence-roberta-hier:
  baseline: 0.667
  concat:   0.673
  improvement: +0.006 (+0.9%) ✓

Ratio: 0.024 / 0.006 = 4.0×
→ Weak encoder benefits 4× more! ✅
```

---

## 📁 File Structure

```
/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/
├── data/embeddings/
│   ├── 4way/
│   │   ├── bert-base-hier/avg_last4/mean/
│   │   ├── bert-base-hier-sentic-concat/avg_last4/mean/
│   │   ├── roberta-base-hier/avg_last4/mean/
│   │   ├── roberta-base-hier-sentic-concat/avg_last4/mean/
│   │   ├── sentence-roberta-hier/avg_last4/mean/
│   │   ├── sentence-roberta-hier-sentic-concat/avg_last4/mean/
│   │   └── senticnet-sentence-level/
│   └── 6way/
│       └── (same structure)
├── results/senticnet_experiments/
│   ├── results.csv                           # All results
│   └── *.json                                # Individual results
├── results/senticnet_analysis/
│   ├── improvements_by_encoder.png
│   ├── complementarity_analysis.png
│   ├── improvements_summary.csv
│   ├── all_results.csv
│   └── ANALYSIS_REPORT.md
└── scripts/
    ├── make_sentic_fusion_hierarchical_npz.py
    ├── run_senticnet_parallel.py
    ├── train_npz_hier_classifier.py
    └── analyze_senticnet_results.py
```

---

## 🎉 After Completion

### If Complementarity CONFIRMED:
```bash
# Proceed with Phase 2 (optional)
# Add gated fusions (alpha=0.10, 0.30)
# Total: 3 × 4 × 2 × 10 = 240 runs
```

### If Complementarity NOT confirmed:
```bash
# Document findings
# Move to Turn-level experiments
```

---

## ⏰ Timeline

| Step | Task | Time |
|------|------|------|
| 1 | Create fusions | 10 min |
| 2 | Verify setup | 5 min |
| 3 | Test run | 2 min |
| 4 | Dry run | 1 min |
| 5 | Full experiment | 2-3 hours |
| 6 | Analyze results | 5 min |
| **TOTAL** | **~3 hours** |

---

🚀 **Ready to launch!** All scripts prepared, just execute step by step! 

**Next:** Fusion → Verify → Launch → Analyze → Report! 🎉
