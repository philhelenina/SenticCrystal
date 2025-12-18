"""
Bayesian Hyperparameter Optimization for SenticCrystal
=====================================================

Multi-objective optimization using Optuna + WandB for:
- Accuracy maximization
- Macro-F1 maximization  
- Weighted-F1 maximization

Uses TPE (Tree-structured Parzen Estimator) to find promising regions.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bayesian_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Saturn Cloud paths
HOME_DIR = Path("/home/jovyan/workspace")
RESULTS_DIR = HOME_DIR / 'results' / 'bayesian_optimization'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def initialize_wandb_optuna(project_name="senticcrystal-bayesian-opt"):
    """Initialize WandB for Bayesian optimization tracking."""
    
    run = wandb.init(
        project=project_name,
        name=f"bayesian_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "optimization_type": "bayesian_hyperparameter_search",
            "config_id": 146,
            "k_value": 2,  # Fixed for hyperparameter search
            "objective": "multi_objective_accuracy_macro_f1_weighted_f1",
            "platform": "Saturn_Cloud_A100",
            "sampler": "TPE"
        },
        tags=["bayesian", "hyperopt", "multi-objective", "a100"]
    )
    
    logger.info("WandB initialized for Bayesian optimization")
    return run

def objective(trial, config_id=146, k_value=2):
    """
    Objective function for Bayesian optimization.
    Multi-objective: maximize accuracy, macro-f1, and weighted-f1
    """
    
    # Define hyperparameter search space
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'hidden_size': trial.suggest_int('hidden_size', 128, 512, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.8),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
        'num_epochs': trial.suggest_int('num_epochs', 50, 200),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 20)
    }
    
    trial_name = f"trial_{trial.number:03d}"
    logger.info(f"🔬 Starting {trial_name} with params: {hyperparams}")
    
    # Log hyperparameters to WandB
    wandb.log({
        "trial_number": trial.number,
        **{f"param_{k}": v for k, v in hyperparams.items()}
    })
    
    start_time = time.time()
    
    # Run experiment with suggested hyperparameters
    cmd = [
        'python', str(HOME_DIR / 'train_turn_classifier.py'),
        '--config_id', str(config_id),
        '--k_value', str(k_value),
        '--model', 'mlp',  # Focus on MLP for hyperparameter search
        '--batch_size', str(hyperparams['batch_size']),
        '--learning_rate', str(hyperparams['learning_rate']),
        '--hidden_size', str(hyperparams['hidden_size']),
        '--dropout_rate', str(hyperparams['dropout_rate']),
        '--weight_decay', str(hyperparams['weight_decay']),
        '--num_epochs', str(hyperparams['num_epochs']),
        '--early_stopping_patience', str(hyperparams['early_stopping_patience'])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # Load results
            result_file = RESULTS_DIR.parent / 'turn_experiments' / f'config{config_id}_k{k_value}_mlp_results.json'
            
            if result_file.exists():
                with open(result_file) as f:
                    exp_data = json.load(f)
                    metrics = exp_data['test_results']
                    
                    accuracy = metrics['accuracy']
                    macro_f1 = metrics['macro_f1']
                    weighted_f1 = metrics['weighted_f1']
                    
                    # Log metrics to WandB
                    wandb.log({
                        f"{trial_name}_accuracy": accuracy,
                        f"{trial_name}_macro_f1": macro_f1,
                        f"{trial_name}_weighted_f1": weighted_f1,
                        f"{trial_name}_execution_time": execution_time,
                        "trial_success": 1
                    })
                    
                    # Store intermediate values for Optuna
                    trial.set_user_attr("accuracy", accuracy)
                    trial.set_user_attr("macro_f1", macro_f1)
                    trial.set_user_attr("weighted_f1", weighted_f1)
                    trial.set_user_attr("execution_time", execution_time)
                    
                    logger.info(f"✅ {trial_name}: Acc={accuracy:.4f}, "
                              f"Macro-F1={macro_f1:.4f}, Weighted-F1={weighted_f1:.4f}, "
                              f"Time={execution_time:.1f}s")
                    
                    # Multi-objective: return weighted combination
                    # Weights: accuracy=0.4, macro_f1=0.3, weighted_f1=0.3
                    combined_score = 0.4 * accuracy + 0.3 * macro_f1 + 0.3 * weighted_f1
                    
                    return combined_score
            
        # Handle failure cases
        logger.error(f"❌ {trial_name} failed: {result.stderr}")
        wandb.log({f"{trial_name}_status": "failed", "trial_success": 0})
        
        # Return poor score for failed trials
        return 0.0
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {trial_name} timed out")
        wandb.log({f"{trial_name}_status": "timeout", "trial_success": 0})
        return 0.0
    
    except Exception as e:
        logger.error(f"💥 {trial_name} crashed: {e}")
        wandb.log({f"{trial_name}_status": "crashed", "trial_success": 0})
        return 0.0

def analyze_optimization_results(study):
    """Analyze and visualize Bayesian optimization results."""
    
    logger.info("="*80)
    logger.info("📊 BAYESIAN OPTIMIZATION ANALYSIS")
    logger.info("="*80)
    
    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    logger.info(f"🏆 Best Trial #{best_trial.number}")
    logger.info(f"🎯 Best Combined Score: {best_value:.4f}")
    logger.info(f"📋 Best Parameters:")
    for param, value in best_params.items():
        logger.info(f"   {param}: {value}")
    
    # Get individual metrics for best trial
    if hasattr(best_trial, 'user_attrs'):
        accuracy = best_trial.user_attrs.get('accuracy', 0)
        macro_f1 = best_trial.user_attrs.get('macro_f1', 0)
        weighted_f1 = best_trial.user_attrs.get('weighted_f1', 0)
        
        logger.info(f"📈 Best Performance:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Macro-F1: {macro_f1:.4f}")
        logger.info(f"   Weighted-F1: {weighted_f1:.4f}")
        
        # Log to WandB
        wandb.log({
            "best_combined_score": best_value,
            "best_accuracy": accuracy,
            "best_macro_f1": macro_f1,
            "best_weighted_f1": weighted_f1,
            "best_trial_number": best_trial.number,
            **{f"best_{k}": v for k, v in best_params.items()}
        })
    
    # Parameter importance analysis
    try:
        importance = optuna.importance.get_param_importances(study)
        logger.info(f"📊 Parameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {param}: {imp:.4f}")
            wandb.log({f"param_importance_{param}": imp})
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")
    
    # Save optimization history
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        }
        if hasattr(trial, 'user_attrs'):
            trial_data.update(trial.user_attrs)
        trials_data.append(trial_data)
    
    # Save results
    optimization_results = {
        'study_info': {
            'n_trials': len(study.trials),
            'best_trial': best_trial.number,
            'best_value': best_value,
            'best_params': best_params
        },
        'trials': trials_data,
        'parameter_importance': importance if 'importance' in locals() else {}
    }
    
    results_file = RESULTS_DIR / f'bayesian_optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    logger.info(f"💾 Optimization results saved: {results_file}")
    wandb.save(str(results_file))
    
    return best_params, optimization_results

def run_validation_experiment(best_params, config_id=146, k_value=2):
    """Run validation experiment with best parameters on LSTM as well."""
    
    logger.info("="*80)
    logger.info("🔬 VALIDATION EXPERIMENT WITH BEST PARAMETERS")
    logger.info("="*80)
    
    validation_results = {}
    
    for model in ['mlp', 'lstm']:
        logger.info(f"🧪 Validating {model.upper()} with best parameters...")
        
        cmd = [
            'python', str(HOME_DIR / 'train_turn_classifier.py'),
            '--config_id', str(config_id),
            '--k_value', str(k_value),
            '--model', model,
            '--batch_size', str(best_params['batch_size']),
            '--learning_rate', str(best_params['learning_rate']),
            '--hidden_size', str(best_params['hidden_size']),
            '--dropout_rate', str(best_params['dropout_rate']),
            '--weight_decay', str(best_params['weight_decay']),
            '--num_epochs', str(best_params['num_epochs']),
            '--early_stopping_patience', str(best_params['early_stopping_patience'])
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                result_file = RESULTS_DIR.parent / 'turn_experiments' / f'config{config_id}_k{k_value}_{model}_results.json'
                
                if result_file.exists():
                    with open(result_file) as f:
                        exp_data = json.load(f)
                        metrics = exp_data['test_results']
                        
                        validation_results[model] = metrics
                        
                        logger.info(f"✅ {model.upper()} Validation:")
                        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
                        logger.info(f"   Macro-F1: {metrics['macro_f1']:.4f}")
                        logger.info(f"   Weighted-F1: {metrics['weighted_f1']:.4f}")
                        
                        wandb.log({
                            f"validation_{model}_accuracy": metrics['accuracy'],
                            f"validation_{model}_macro_f1": metrics['macro_f1'],
                            f"validation_{model}_weighted_f1": metrics['weighted_f1']
                        })
            
        except Exception as e:
            logger.error(f"❌ {model} validation failed: {e}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter optimization")
    parser.add_argument('--config_id', type=int, default=146, help='Configuration ID')
    parser.add_argument('--k_value', type=int, default=2, help='K value for optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--wandb_project', type=str, default='senticcrystal-bayesian-opt',
                       help='WandB project name')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Optuna study name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Initialize WandB
    wandb_run = initialize_wandb_optuna(args.wandb_project)
    
    # Create study name
    if args.study_name is None:
        args.study_name = f"senticcrystal_bayesian_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    start_time = time.time()
    logger.info("="*100)
    logger.info("🚀 STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION")
    logger.info("="*100)
    logger.info(f"Config ID: {args.config_id}")
    logger.info(f"K Value: {args.k_value}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=args.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # WandB callback for Optuna integration
        wandbc = WeightsAndBiasesCallback(
            metric_name="combined_score",
            wandb_kwargs={"project": args.wandb_project}
        )
        
        # Run optimization
        logger.info(f"🔍 Starting {args.n_trials} trials of Bayesian optimization...")
        study.optimize(
            lambda trial: objective(trial, args.config_id, args.k_value),
            n_trials=args.n_trials,
            callbacks=[wandbc],
            show_progress_bar=True
        )
        
        # Analyze results
        best_params, optimization_results = analyze_optimization_results(study)
        
        # Run validation experiments
        validation_results = run_validation_experiment(best_params, args.config_id, args.k_value)
        
        total_time = time.time() - start_time
        logger.info("="*100)
        logger.info("🎉 BAYESIAN OPTIMIZATION COMPLETED SUCCESSFULLY")
        logger.info("="*100)
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"End time: {datetime.now()}")
        logger.info(f"Best parameters found: {best_params}")
        
        wandb.log({
            "optimization_total_time": total_time,
            "optimization_status": "completed_successfully",
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        })
        
        return best_params
        
    except Exception as e:
        logger.error(f"💥 Optimization failed: {e}")
        wandb.log({"optimization_status": "failed", "error": str(e)})
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    best_hyperparams = main()
    print(f"\n🏆 Best hyperparameters: {best_hyperparams}")

# Usage examples:
# python bayesian_hyperparameter_optimization.py --n_trials 50
# python bayesian_hyperparameter_optimization.py --n_trials 100 --k_value 2
# python bayesian_hyperparameter_optimization.py --n_trials 30 --config_id 146