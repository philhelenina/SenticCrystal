# Complete Experimental Suite (n=10 seeds)

## 📋 Overview

All experiments with **10 seeds (42-51)** for robust statistical analysis.

### Total Experiments: 1,200
- **Flat baseline**: 720 experiments (3 encoders × 2 layers × 3 pools × 2 classifiers × 2 tasks × 10 seeds)
- **Hierarchical**: 480 experiments (2 layers × 2 pools × 5 aggregators × 2 classifiers × 2 tasks × 10 seeds)

### Estimated Time
- **Flat**: ~45-50 hours total (~12 hours per GPU)
- **Hierarchical**: ~35-40 hours total (~10 hours per GPU)
- **Total**: ~85 hours if run serially, **~22 hours if parallel** (4 GPUs)

---

## 🗂️ Directory Structure (New & Clean!)

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
│   ├── hierarchical/
│   │   └── sentence-roberta-hier/
│   │       ├── last/
│   │       │   ├── mean/
│   │       │   │   ├── mean/
│   │       │   │   │   ├── mlp/
│   │       │   │   │   │   ├── seed_42/results.json
│   │       │   │   │   │   └── ...
│   │       │   │   │   └── lstm/
│   │       │   │   ├── attn/
│   │       │   │   ├── sum/
│   │       │   │   ├── expdecay/
│   │       │   │   └── lstm/
│   │       │   └── wmean_pos_rev/
│   │       └── avg_last4/
│   └── summary/
│       ├── gpu0_mlp_summary.txt
│       ├── gpu1_lstm_summary.txt
│       ├── hier_gpu0_4way_summary.txt
│       └── ...
└── 6way/
    └── (same structure)
```

**Key difference from old structure:**
- Old: `results/baseline/` and `results/hier_baseline/`
- New: `results_n10/` with clean `flat/` and `hierarchical/` separation
- **No overlap** with previous results!

---

## 📦 Script Files

### Flat Baseline (4 scripts):
1. `run_n10_gpu0_4way_mlp.sh` - GPU 0: 4way MLP (180 runs)
2. `run_n10_gpu1_4way_lstm.sh` - GPU 1: 4way LSTM (180 runs)
3. `run_n10_gpu2_6way_mlp.sh` - GPU 2: 6way MLP (180 runs)
4. `run_n10_gpu3_6way_lstm.sh` - GPU 3: 6way LSTM (180 runs)

### Hierarchical (4 scripts):
5. `run_n10_hier_gpu0_4way.sh` - GPU 0: 4way (mean, lstm, attn-mlp)
6. `run_n10_hier_gpu1_4way.sh` - GPU 1: 4way (attn-lstm, expdecay, sum)
7. `run_n10_hier_gpu2_6way.sh` - GPU 2: 6way (mean, lstm, attn-mlp)
8. `run_n10_hier_gpu3_6way.sh` - GPU 3: 6way (attn-lstm, expdecay, sum)

### Master Scripts (2):
9. `run_all_n10_flat.sh` - Launch all flat experiments
10. `run_all_n10_hier.sh` - Launch all hierarchical experiments

---

## 🚀 Quick Start

### Step 1: Edit Paths (ONE TIME ONLY!)

Open each script and update the `BASE_DIR` variable:

```bash
# Edit this line in EACH script:
BASE_DIR="/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment"
```

**Files to edit** (just change one line in each):
- run_n10_gpu0_4way_mlp.sh
- run_n10_gpu1_4way_lstm.sh
- run_n10_gpu2_6way_mlp.sh
- run_n10_gpu3_6way_lstm.sh
- run_n10_hier_gpu0_4way.sh
- run_n10_hier_gpu1_4way.sh
- run_n10_hier_gpu2_6way.sh
- run_n10_hier_gpu3_6way.sh

### Step 2: Make Scripts Executable

```bash
chmod +x run_*.sh
```

### Step 3: Run Experiments!

#### Option A: Run Everything (Flat + Hier)
```bash
# Run in background
nohup bash run_all_n10_flat.sh &
nohup bash run_all_n10_hier.sh &

# Or run them sequentially
bash run_all_n10_flat.sh && bash run_all_n10_hier.sh
```

#### Option B: Run Separately

**Flat only:**
```bash
nohup bash run_all_n10_flat.sh > flat_n10.log 2>&1 &
```

**Hierarchical only:**
```bash
nohup bash run_all_n10_hier.sh > hier_n10.log 2>&1 &
```

#### Option C: Run Individual GPUs

**If you want more control:**
```bash
# Run one GPU at a time
bash run_n10_gpu0_4way_mlp.sh
bash run_n10_gpu1_4way_lstm.sh
# etc...
```

---

## 📊 Monitoring Progress

### Check logs:
```bash
# Flat experiments
tail -f n10_gpu0_flat.log
tail -f n10_gpu1_flat.log
tail -f n10_gpu2_flat.log
tail -f n10_gpu3_flat.log

# Hierarchical experiments
tail -f n10_gpu0_hier.log
tail -f n10_gpu1_hier.log
tail -f n10_gpu2_hier.log
tail -f n10_gpu3_hier.log
```

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Check summary files:
```bash
# While running
tail -f results_n10/4way/summary/gpu0_mlp_summary.txt

# After completion
cat results_n10/4way/summary/*.txt
```

---

## 📈 Expected Timeline

### Per GPU (approximate):
```
Flat (180 runs × 8 min):     ~24 hours
Hierarchical (120 runs × 5 min): ~10 hours
```

### Parallel (all 4 GPUs):
```
Day 1: Flat experiments (24 hours)
Day 2: Hierarchical experiments (10 hours)
Total: ~34 hours (~1.5 days)
```

**Recommendation:** Start before leaving for the day, check next morning!

---

## 🐛 Troubleshooting

### Script fails immediately:
```bash
# Check if python script exists
ls train_npz_classifier_4way.py
ls train_npz_hier_classifier_4way.py

# Check if embeddings exist
ls -la data/embeddings/4way/sentence-roberta/
```

### GPU out of memory:
```bash
# Reduce batch size in scripts
# Change BS=64 to BS=32
```

### Missing results.json:
```bash
# Check if experiment actually ran
tail -100 n10_gpu0_flat.log

# Check output directory
ls results_n10/4way/flat/sentence-roberta/last/mean/mlp/seed_42/
```

### Disk space issues:
```bash
# Check disk space
df -h

# Each results.json is ~5KB
# 1,200 experiments × 5KB = ~6MB (very small!)
```

---

## ✅ Verification Checklist

After experiments complete:

### Check total number of results files:
```bash
# Should be 720 for flat
find results_n10/4way/flat -name "results.json" | wc -l
find results_n10/6way/flat -name "results.json" | wc -l

# Should be 480 for hierarchical
find results_n10/4way/hierarchical -name "results.json" | wc -l
find results_n10/6way/hierarchical -name "results.json" | wc -l
```

### Check for any failures:
```bash
# Search logs for "Failed"
grep -i "failed" *.log

# Check summary files
grep "❌" results_n10/*/summary/*.txt
```

### Verify seeds:
```bash
# Should have seeds 42-51 for each config
ls results_n10/4way/flat/sentence-roberta/last/mean/mlp/
# Expected: seed_42/ seed_43/ ... seed_51/
```

---

## 📦 Collecting Results

After all experiments complete:

```python
import json
import pandas as pd
from pathlib import Path

results = []
base_dir = Path("results_n10")

for results_file in base_dir.rglob("results.json"):
    with open(results_file) as f:
        data = json.load(f)
    
    # Parse path to extract config
    parts = results_file.parts
    # Example: results_n10/4way/flat/sentence-roberta/last/mean/mlp/seed_42/results.json
    
    task = parts[1]  # 4way or 6way
    arch_type = parts[2]  # flat or hierarchical
    
    if arch_type == "flat":
        encoder = parts[3]
        layer = parts[4]
        pool = parts[5]
        classifier = parts[6]
        seed = parts[7].replace("seed_", "")
        aggregator = None
    else:  # hierarchical
        encoder = parts[3]
        layer = parts[4]
        pool = parts[5]
        aggregator = parts[6]
        classifier = parts[7]
        seed = parts[8].replace("seed_", "")
    
    results.append({
        "task": task,
        "type": arch_type,
        "encoder": encoder,
        "layer": layer,
        "pool": pool,
        "aggregator": aggregator,
        "classifier": classifier,
        "seed": int(seed),
        "accuracy": data["metrics"]["accuracy"],
        "macro_f1": data["metrics"]["macro_f1"],
        "weighted_f1": data["metrics"]["weighted_f1"]
    })

df = pd.DataFrame(results)
df.to_csv("all_results_n10.csv", index=False)
print(f"Collected {len(df)} results")
print(f"Seeds: {sorted(df['seed'].unique())}")
```

---

## 📊 Statistical Analysis (After Collection)

```python
import pandas as pd
from scipy import stats

df = pd.read_csv("all_results_n10.csv")

# Example: Compare architectures (4-way, n=10)
flat_best = df[
    (df['task'] == '4way') &
    (df['type'] == 'flat') &
    (df['encoder'] == 'sentence-roberta') &
    (df['layer'] == 'avg_last4') &
    (df['pool'] == 'mean') &
    (df['classifier'] == 'lstm')
]['weighted_f1'].values

hier_best = df[
    (df['task'] == '4way') &
    (df['type'] == 'hierarchical') &
    (df['encoder'] == 'sentence-roberta-hier') &
    (df['layer'] == 'avg_last4') &
    (df['pool'] == 'wmean_pos_rev') &
    (df['aggregator'] == 'mean') &
    (df['classifier'] == 'mlp')
]['weighted_f1'].values

print(f"Flat:  {flat_best.mean()*100:.2f}% ± {flat_best.std()*100:.2f}%")
print(f"Hier:  {hier_best.mean()*100:.2f}% ± {hier_best.std()*100:.2f}%")

u, p = stats.mannwhitneyu(hier_best, flat_best, alternative='greater')
print(f"p-value: {p:.4f}")
```

---

## 🎉 Success Criteria

### You're done when:
- ✅ All 1,200 results.json files exist
- ✅ No failures in summary files
- ✅ Each config has exactly 10 seeds (42-51)
- ✅ Results collected into CSV
- ✅ Statistical analysis shows significance (p<0.05)

**Then you can write a killer paper!** 📝🚀

---

## 💾 Backup

After completion, backup your results:

```bash
# Compress results
tar -czf results_n10_backup.tar.gz results_n10/

# Copy to safe location
cp results_n10_backup.tar.gz /path/to/backup/

# Verify
tar -tzf results_n10_backup.tar.gz | head
```

---

## 🆘 Need Help?

Common issues:

1. **"Script not found"** → Check SCRIPT_DIR path
2. **"Embedding not found"** → Check EMB_BASE path
3. **"Permission denied"** → Run `chmod +x run_*.sh`
4. **"CUDA out of memory"** → Reduce batch_size
5. **Results incomplete** → Check logs for specific errors

Good luck! 🍀
