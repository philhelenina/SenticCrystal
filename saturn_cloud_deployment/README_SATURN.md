# Saturn Cloud A100 Deployment for SenticCrystal

## Quick Setup

1. **Create conda environment (mamba faster):**
```bash
# Option 1: Automatic setup
bash setup_saturn_environment.sh

# Option 2: Manual setup
mamba env create -f environment.yml
conda activate senticcrystal
```

2. **Upload your IEMOCAP data to `/home/jovyan/workspace/data/iemocap_4way_data/`**

3. **Generate embeddings (Config146):**
```bash
python embeddings_saturn.py --config 146 --train --val --test
```

4. **Run experiments:**
```bash
# Full experiments
python run_comprehensive_experiments.py --config_id 146

# Test with K=2 only
python run_comprehensive_experiments.py --config_id 146 --turn_only --k_values 2
```

## File Structure
```
/home/jovyan/workspace/
├── data/iemocap_4way_data/          # Your IEMOCAP CSV files
├── wordnet/                         # WordNet-Affect data
├── src/                            # Source modules
├── configs/                        # Configuration files
├── scripts/embeddings/             # Generated embeddings
└── results/                        # Experiment results
```

## GPU Monitoring
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## Expected Results
- **K=0 (Baseline)**: ~46% accuracy
- **K=2**: ~66% accuracy  
- **K=4,6**: Testing on Saturn Cloud A100

All results saved to `/results/` with comprehensive metrics and confusion matrices.