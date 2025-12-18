"""
Comprehensive Turn Experiments Runner
===================================

Run complete turn experiments for K=0,2,4,6 with both LSTM and MLP models.
Designed for Saturn Cloud A100 deployment with comprehensive logging.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
RESULTS_DIR = HOME_DIR / 'results'

def run_baseline_experiments(config_id):
    """Run baseline experiments (K=0)."""
    logger.info("="*60)
    logger.info("RUNNING BASELINE EXPERIMENTS (K=0)")
    logger.info("="*60)
    
    cmd = [
        'python', str(HOME_DIR / 'scripts' / 'train_baseline_classifier.py'),
        '--config_id', str(config_id),
        '--model', 'both',
        '--batch_size', '32',
        '--num_epochs', '300',
        '--early_stopping_patience', '10'
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"Baseline experiment failed: {result.stderr}")
            return False
        else:
            logger.info("Baseline experiments completed successfully")
            logger.info(f"Baseline execution time: {time.time() - start_time:.2f} seconds")
            return True
    except subprocess.TimeoutExpired:
        logger.error("Baseline experiment timed out")
        return False

def run_turn_experiments(config_id, k_values, models=['lstm', 'mlp']):
    """Run turn experiments for specified K values."""
    logger.info("="*60)
    logger.info("RUNNING TURN EXPERIMENTS")
    logger.info("="*60)
    
    results = {}
    
    for k in k_values:
        logger.info(f"\n{'='*40}")
        logger.info(f"RUNNING K={k} EXPERIMENTS")
        logger.info(f"{'='*40}")
        
        for model in models:
            logger.info(f"\nRunning K={k}, Model={model}")
            
            cmd = [
                'python', str(HOME_DIR / 'scripts' / 'train_turn_classifier.py'),
                '--config_id', str(config_id),
                '--k_value', str(k),
                '--model', model,
                '--batch_size', '32',
                '--num_epochs', '100',
                '--early_stopping_patience', '10'
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
                execution_time = time.time() - start_time
                
                if result.returncode != 0:
                    logger.error(f"K={k}, Model={model} failed: {result.stderr}")
                    results[f'k{k}_{model}'] = {'status': 'failed', 'error': result.stderr}
                else:
                    logger.info(f"K={k}, Model={model} completed successfully")
                    logger.info(f"Execution time: {execution_time:.2f} seconds")
                    results[f'k{k}_{model}'] = {'status': 'success', 'execution_time': execution_time}
                    
            except subprocess.TimeoutExpired:
                logger.error(f"K={k}, Model={model} timed out")
                results[f'k{k}_{model}'] = {'status': 'timeout'}
    
    return results

def collect_all_results(config_id):
    """Collect and summarize all experimental results."""
    logger.info("="*60)
    logger.info("COLLECTING AND SUMMARIZING RESULTS")
    logger.info("="*60)
    
    baseline_dir = RESULTS_DIR / 'baseline_classifiers'
    turn_dir = RESULTS_DIR / 'turn_experiments'
    
    summary = {
        'config_id': config_id,
        'experiment_date': datetime.now().isoformat(),
        'baseline_results': {},
        'turn_results': {}
    }
    
    # Collect baseline results
    baseline_file = baseline_dir / f'config{config_id}_baseline_comparison.json'
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)
            for model_type in ['mlp', 'lstm']:
                if model_type in baseline_data:
                    test_res = baseline_data[model_type]['test_results']
                    summary['baseline_results'][f'{model_type}_k0'] = {
                        'accuracy': test_res['accuracy'],
                        'macro_f1': test_res['macro_f1'],
                        'weighted_f1': test_res['weighted_f1']
                    }
    
    # Collect turn experiment results
    k_values = [2, 4, 6]
    models = ['lstm', 'mlp']
    
    for k in k_values:
        for model in models:
            result_file = turn_dir / f'config{config_id}_k{k}_{model}_results.json'
            if result_file.exists():
                with open(result_file) as f:
                    turn_data = json.load(f)
                    test_res = turn_data['test_results']
                    summary['turn_results'][f'{model}_k{k}'] = {
                        'accuracy': test_res['accuracy'],
                        'macro_f1': test_res['macro_f1'],
                        'weighted_f1': test_res['weighted_f1']
                    }
    
    # Save comprehensive summary
    summary_file = RESULTS_DIR / f'config{config_id}_comprehensive_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Comprehensive summary saved: {summary_file}")
    
    # Print performance comparison table
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Experiment':<15} {'Accuracy':<10} {'Macro-F1':<10} {'Weighted-F1':<12}")
    logger.info("-" * 80)
    
    # Baseline results
    for exp_name, metrics in summary['baseline_results'].items():
        logger.info(f"{exp_name:<15} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} {metrics['weighted_f1']:<12.4f}")
    
    # Turn experiment results
    for exp_name, metrics in summary['turn_results'].items():
        logger.info(f"{exp_name:<15} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} {metrics['weighted_f1']:<12.4f}")
    
    logger.info("="*80)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive turn experiments")
    parser.add_argument('--config_id', type=int, default=146, help='Configuration ID')
    parser.add_argument('--k_values', nargs='+', type=int, default=[2, 4, 6], 
                       help='K values for turn experiments')
    parser.add_argument('--models', nargs='+', choices=['lstm', 'mlp'], default=['lstm', 'mlp'],
                       help='Models to train')
    parser.add_argument('--skip_baseline', action='store_true', 
                       help='Skip baseline experiments if already completed')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only baseline experiments')
    parser.add_argument('--turn_only', action='store_true',
                       help='Run only turn experiments (skip baseline)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE TURN EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"Config ID: {args.config_id}")
    logger.info(f"K values: {args.k_values}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Start time: {datetime.now()}")
    
    success = True
    
    # Run baseline experiments
    if not args.turn_only and not args.baseline_only:
        if not args.skip_baseline:
            if not run_baseline_experiments(args.config_id):
                logger.error("Baseline experiments failed. Stopping.")
                return 1
    elif args.baseline_only:
        if not run_baseline_experiments(args.config_id):
            logger.error("Baseline experiments failed.")
            return 1
        logger.info("Baseline-only mode completed.")
        return 0
    
    # Run turn experiments
    if not args.baseline_only:
        turn_results = run_turn_experiments(args.config_id, args.k_values, args.models)
        
        # Check if any experiments failed
        failed_experiments = [exp for exp, result in turn_results.items() 
                            if result.get('status') != 'success']
        
        if failed_experiments:
            logger.warning(f"Some experiments failed: {failed_experiments}")
            success = False
    
    # Collect and summarize results
    if not args.baseline_only and not args.turn_only:
        summary = collect_all_results(args.config_id)
    
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info("COMPREHENSIVE EXPERIMENTS COMPLETED")
    logger.info("="*80)
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Overall status: {'SUCCESS' if success else 'PARTIAL SUCCESS'}")
    
    return 0 if success else 1





if __name__ == "__main__":
    exit(main())

# Usage examples:
# python run_comprehensive_experiments.py --config_id 146
# python run_comprehensive_experiments.py --config_id 146 --k_values 2 4 --models lstm
# python run_comprehensive_experiments.py --config_id 146 --skip_baseline
# python run_comprehensive_experiments.py --config_id 146 --baseline_only
# python run_comprehensive_experiments.py --config_id 146 --turn_only --k_values 2