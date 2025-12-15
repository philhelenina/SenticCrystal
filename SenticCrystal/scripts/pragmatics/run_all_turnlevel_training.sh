#!/bin/bash
###############################################################################
# Run All Turn-Level K-Sweep Training - 4 GPU PARALLEL VERSION
# 
# Optimized for: 4x NVIDIA A100-SXM4-80GB
# 
# Strategy:
#   - 1,230 total jobs distributed across 4 GPUs
#   - Each GPU runs jobs sequentially
#   - ~308 jobs per GPU
#   - Estimated time: ~10-12 hours
#
# Total jobs: 1,230 (3 configs × 2 tasks × 5 seeds × 41 K-values)
###############################################################################

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOME_DIR="/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_turnlevel_k_sweep.py"
LOG_DIR="${HOME_DIR}/results/turnlevel_k_sweep/logs"

# GPU configuration
NUM_GPUS=4  # 4x A100

# Check if training script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Training script not found: $PYTHON_SCRIPT"
    echo "   Expected: $PYTHON_SCRIPT"
    echo ""
    echo "   Please create this script or check the path!"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# K-sweep parameters
K_START=0
K_END=200
K_STEP=5

# Experimental parameters
CONFIGS=("baseline" "no_position_pooling" "no_lexical_fusion")
TASKS=("4way" "6way")
SEEDS=(42 43 44 45 46)

# Generate all jobs
K_VALUES=()
for k in $(seq $K_START $K_STEP $K_END); do
    K_VALUES+=($k)
done

# Calculate total jobs
TOTAL_JOBS=$((${#CONFIGS[@]} * ${#TASKS[@]} * ${#SEEDS[@]} * ${#K_VALUES[@]}))

echo "=================================="
echo "K-SWEEP TRAINING - 4 GPU PARALLEL"
echo "=================================="
echo "Environment: 4x NVIDIA A100-SXM4-80GB"
echo ""
echo "K range: ${K_START}~${K_END}, step=${K_STEP}"
echo "K values: ${#K_VALUES[@]}"
echo "Configs: ${CONFIGS[@]}"
echo "Tasks: ${TASKS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Total jobs: ${TOTAL_JOBS}"
echo "Jobs per GPU: ~$((TOTAL_JOBS / NUM_GPUS))"
echo "Estimated time: ~10-12 hours"
echo ""
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Check if results already exist
existing_count=0
for config in "${CONFIGS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for K in "${K_VALUES[@]}"; do
                result_dir="${HOME_DIR}/results/turnlevel_k_sweep/${config}/${task}/seed_${seed}/K_${K}"
                if [ -f "${result_dir}/metrics.json" ]; then
                    ((existing_count++))
                fi
            done
        done
    done
done

if [ $existing_count -gt 0 ]; then
    echo "⚠️  Found ${existing_count} existing results"
    echo ""
    read -p "Skip existing? (y=skip, n=overwrite): " -n 1 -r
    echo
    SKIP_EXISTING=$REPLY
else
    SKIP_EXISTING="n"
fi

echo ""
read -p "Start training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Function to run jobs on a specific GPU
run_gpu_jobs() {
    local gpu_id=$1
    local job_list_file=$2
    local log_file="${LOG_DIR}/gpu_${gpu_id}.log"
    
    echo "GPU ${gpu_id}: Starting..." | tee -a "$log_file"
    
    local job_count=0
    local gpu_start_time=$(date +%s)
    
    while IFS='|' read -r config task seed K; do
        ((job_count++))
        
        # Check if should skip
        if [[ "$SKIP_EXISTING" =~ ^[Yy]$ ]]; then
            result_dir="${HOME_DIR}/results/turnlevel_k_sweep/${config}/${task}/seed_${seed}/K_${K}"
            if [ -f "${result_dir}/metrics.json" ]; then
                echo "GPU ${gpu_id} [${job_count}]: SKIP ${config}/${task}/seed_${seed}/K_${K}" | tee -a "$log_file"
                continue
            fi
        fi
        
        echo "GPU ${gpu_id} [${job_count}]: ${config}/${task}/seed_${seed}/K_${K}" | tee -a "$log_file"
        
        # Run training
        CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT" \
            --config "$config" \
            --task "$task" \
            --seed "$seed" \
            --K "$K" \
            >> "$log_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "GPU ${gpu_id} [${job_count}]: ✓ Success" | tee -a "$log_file"
        else
            echo "GPU ${gpu_id} [${job_count}]: ✗ FAILED" | tee -a "$log_file"
        fi
        
        # Show progress every 10 jobs
        if [ $((job_count % 10)) -eq 0 ]; then
            elapsed=$(($(date +%s) - gpu_start_time))
            avg_time=$((elapsed / job_count))
            echo "GPU ${gpu_id}: Progress ${job_count} jobs, avg ${avg_time}s/job" | tee -a "$log_file"
        fi
        
    done < "$job_list_file"
    
    local gpu_end_time=$(date +%s)
    local gpu_total_time=$((gpu_end_time - gpu_start_time))
    
    echo "GPU ${gpu_id}: ✅ COMPLETED ${job_count} jobs in $((gpu_total_time / 3600))h $((gpu_total_time % 3600 / 60))m" | tee -a "$log_file"
}

# Generate job lists for each GPU
echo "Distributing jobs across ${NUM_GPUS} GPUs..."

job_idx=0
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    > "${LOG_DIR}/jobs_gpu_${gpu_id}.txt"  # Clear file
done

for config in "${CONFIGS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for K in "${K_VALUES[@]}"; do
                gpu_id=$((job_idx % NUM_GPUS))
                echo "${config}|${task}|${seed}|${K}" >> "${LOG_DIR}/jobs_gpu_${gpu_id}.txt"
                ((job_idx++))
            done
        done
    done
done

# Show distribution
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    job_count=$(wc -l < "${LOG_DIR}/jobs_gpu_${gpu_id}.txt")
    echo "GPU ${gpu_id}: ${job_count} jobs"
done

echo ""
echo "Starting parallel execution..."
echo "Monitor progress: tail -f ${LOG_DIR}/gpu_*.log"
echo ""

# Start all GPUs in parallel
start_time=$(date +%s)

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    run_gpu_jobs $gpu_id "${LOG_DIR}/jobs_gpu_${gpu_id}.txt" &
done

# Wait for all GPUs to finish
wait

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "=================================="
echo "✅ ALL JOBS COMPLETED!"
echo "=================================="
echo "Total time: $((total_time / 3600))h $((total_time % 3600 / 60))m"
echo "Jobs completed: ${TOTAL_JOBS}"
echo ""
echo "📊 Check results:"
echo "   ls results/turnlevel_k_sweep/baseline/4way/seed_42/ | wc -l"
echo "   Should be: ${#K_VALUES[@]} (one directory per K value)"
echo ""
echo "📋 Next step:"
echo "   python aggregate_turnlevel_results_stratified.py"
