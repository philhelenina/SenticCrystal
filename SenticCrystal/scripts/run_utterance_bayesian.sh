#!/bin/bash
# Utterance-level Bayesian Hyperparameter Optimization
#
# Optimizes hyperparameters for:
#   - Flat (sentence-roberta): mean, wmean_pos, wmean_pos_rev
#   - Hier (sentence-roberta-hier): mean, wmean_pos, wmean_pos_rev
#
# Uses 3 seeds (42, 43, 44) average for robust optimization
# 30 trials per configuration

source ~/miniconda3/etc/profile.d/conda.sh
conda activate senticcrystal

cd /home/cheonkaj/projects/SenticCrystal/scripts

SCRIPT="bayesian_flat_optimization.py"
N_TRIALS=30
SEEDS="42 43 44"

mkdir -p bayesian_logs

echo "======================================"
echo "UTTERANCE-LEVEL BAYESIAN OPTIMIZATION"
echo "======================================"
echo "Trials: $N_TRIALS"
echo "Seeds: $SEEDS"
echo "Start: $(date)"
echo ""

# Function to run optimization
run_opt() {
    local task=$1
    local model_tag=$2
    local pool=$3
    local gpu=$4

    local model_type="flat"
    if [ "$model_tag" == "sentence-roberta-hier" ]; then
        model_type="hier"
    fi

    echo "[${task}|${model_type}|${pool}] Starting on GPU ${gpu}..."

    CUDA_VISIBLE_DEVICES=${gpu} python ${SCRIPT} \
        --task ${task} \
        --model_tag ${model_tag} \
        --pool ${pool} \
        --n_trials ${N_TRIALS} \
        --seeds ${SEEDS} \
        --gpu 0 \
        > bayesian_logs/utterance_${task}_${model_type}_${pool}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[${task}|${model_type}|${pool}] Done"
    else
        echo "[${task}|${model_type}|${pool}] FAILED"
    fi
}

# ============================================
# FLAT (sentence-roberta)
# ============================================
echo ""
echo "=== FLAT OPTIMIZATION ==="

# mean
echo ""
echo "--- Flat + mean ---"
run_opt "4way" "sentence-roberta" "mean" 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "mean" 1 &
PID2=$!
wait $PID1 $PID2

# wmean_pos
echo ""
echo "--- Flat + wmean_pos ---"
run_opt "4way" "sentence-roberta" "wmean_pos" 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "wmean_pos" 1 &
PID2=$!
wait $PID1 $PID2

# wmean_pos_rev
echo ""
echo "--- Flat + wmean_pos_rev ---"
run_opt "4way" "sentence-roberta" "wmean_pos_rev" 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "wmean_pos_rev" 1 &
PID2=$!
wait $PID1 $PID2

# ============================================
# HIER (sentence-roberta-hier)
# ============================================
echo ""
echo "=== HIER OPTIMIZATION ==="

# mean
echo ""
echo "--- Hier + mean ---"
run_opt "4way" "sentence-roberta-hier" "mean" 0 &
PID1=$!
run_opt "6way" "sentence-roberta-hier" "mean" 1 &
PID2=$!
wait $PID1 $PID2

# wmean_pos
echo ""
echo "--- Hier + wmean_pos ---"
run_opt "4way" "sentence-roberta-hier" "wmean_pos" 0 &
PID1=$!
run_opt "6way" "sentence-roberta-hier" "wmean_pos" 1 &
PID2=$!
wait $PID1 $PID2

# wmean_pos_rev
echo ""
echo "--- Hier + wmean_pos_rev ---"
run_opt "4way" "sentence-roberta-hier" "wmean_pos_rev" 0 &
PID1=$!
run_opt "6way" "sentence-roberta-hier" "wmean_pos_rev" 1 &
PID2=$!
wait $PID1 $PID2

echo ""
echo "======================================"
echo "ALL UTTERANCE-LEVEL OPTIMIZATION DONE!"
echo "End: $(date)"
echo "======================================"
echo ""
echo "Results saved in: results/bayesian_optimization/"
echo ""
echo "Next step: Run turn-level experiments with optimized params"
echo "  - Update train_turnlevel_k_sweep_bayesian.py to load new params"
echo "  - Re-run turn-level K-sweep with optimized hyperparameters"
