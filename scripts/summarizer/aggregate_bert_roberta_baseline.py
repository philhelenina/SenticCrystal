#!/usr/bin/env python3
"""
aggregate_bert_roberta_baseline.py

Aggregate baseline results for BERT-base-hier and RoBERTa-base-hier
from results_n10/{4way,6way}/hierarchical/{encoder}/avg_last4/mean/mean/mlp/seed_XX/
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

# Base directory
BASE_DIR = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RESULTS_BASE = BASE_DIR / "results_n10"

def find_all_results():
    """Find all results.json files"""
    results = []
    
    # Tasks and encoders
    tasks = ["4way", "6way"]
    encoders = ["bert-base-hier", "roberta-base-hier"]
    
    for task in tasks:
        for encoder in encoders:
            # Path: results_n10/{task}/hierarchical/{encoder}/avg_last4/mean/mean/mlp/
            pattern = RESULTS_BASE / task / "hierarchical" / encoder / "avg_last4" / "mean" / "mean" / "mlp"
            
            if not pattern.exists():
                print(f"[SKIP] Path not found: {pattern}")
                continue
            
            # Find all JSON files matching pattern: {encoder}_{task}_seed*.json
            json_pattern = f"{encoder}_{task}_seed*.json"
            json_files = list(pattern.glob(json_pattern))
            
            if len(json_files) == 0:
                print(f"[WARNING] No JSON files found in: {pattern}")
                print(f"  Looking for pattern: {json_pattern}")
            else:
                print(f"[INFO] Found {len(json_files)} files in {pattern}")
            
            results.extend(json_files)
    
    return results

def parse_result_file(json_path: Path):
    """Parse a single results JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract seed from filename: {encoder}_{task}_seed{X}.json
        filename = json_path.name
        seed_match = filename.split('_seed')[-1].replace('.json', '')
        try:
            seed = int(seed_match)
        except:
            seed = -1
        
        # Extract info
        result = {
            'type': 'hierarchical',
            'task': data.get('task', ''),
            'encoder': data.get('encoder', ''),
            'layer': data.get('layer', ''),
            'pool': data.get('pool', ''),
            'aggregator': 'mean',  # Fixed for baseline
            'classifier': data.get('classifier', ''),
            'seed': seed,
            
            # Metrics
            'accuracy': data.get('test_acc', 0.0),
            'macro_f1': data.get('test_f1_macro', 0.0),
            'weighted_f1': data.get('test_f1_weighted', 0.0),
            
            # Additional info
            'best_val_acc': data.get('best_val_acc', 0.0),
            'best_epoch': data.get('best_epoch', -1),
            'stopped_early': data.get('stopped_early', False),
            
            # Path
            'path': str(json_path)
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to parse {json_path}: {e}")
        return None

def main():
    print("="*80)
    print("AGGREGATE BERT/ROBERTA BASELINE RESULTS")
    print("="*80)
    print()
    
    # Find all results
    print("Finding results.json files...")
    result_files = find_all_results()
    print(f"Found {len(result_files)} result files")
    print()
    
    # Parse all files
    print("Parsing results...")
    results = []
    for json_path in result_files:
        result = parse_result_file(json_path)
        if result:
            results.append(result)
    
    print(f"Successfully parsed {len(results)} files")
    print()
    
    if len(results) == 0:
        print("[ERROR] No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by encoder, task, seed
    df = df.sort_values(['encoder', 'task', 'seed'])
    
    # Save raw results
    output_path = BASE_DIR / "results_n10" / "bert_roberta_baseline_all.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved all results to: {output_path}")
    print()
    
    # Compute statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    group_cols = ['task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier']
    stats = df.groupby(group_cols).agg({
        'accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'macro_f1': ['mean', 'std', 'min', 'max'],
        'weighted_f1': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
    
    # Sort by weighted_f1_mean descending
    stats = stats.sort_values('weighted_f1_mean', ascending=False)
    
    # Save summary
    summary_path = BASE_DIR / "results_n10" / "bert_roberta_baseline_summary.csv"
    stats.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to: {summary_path}")
    print()
    
    # Display summary
    print("Results by encoder and task:")
    print()
    for _, row in stats.iterrows():
        print(f"{row['encoder']:25s} | {row['task']:5s} | {row['aggregator']:10s} | {row['classifier']:4s}")
        print(f"  Weighted F1: {row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f} "
              f"(n={int(row['accuracy_count'])}, max={row['weighted_f1_max']:.4f})")
        print(f"  Macro F1:    {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}")
        print(f"  Accuracy:    {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print()
    
    # Best runs per task
    print("="*80)
    print("BEST RUNS")
    print("="*80)
    print()
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        best_idx = task_df['weighted_f1'].idxmax()
        best = task_df.loc[best_idx]
        
        print(f"{task.upper()}:")
        print(f"  Encoder:     {best['encoder']}")
        print(f"  Seed:        {best['seed']}")
        print(f"  Accuracy:    {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
        print(f"  Macro F1:    {best['macro_f1']:.4f}")
        print(f"  Weighted F1: {best['weighted_f1']:.4f} ({best['weighted_f1']*100:.2f}%)")
        print()
    
    # Compare encoders
    print("="*80)
    print("ENCODER COMPARISON")
    print("="*80)
    print()
    
    comparison = df.groupby(['task', 'encoder']).agg({
        'weighted_f1': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    print(comparison)
    print()
    
    print("="*80)
    print("DONE")
    print("="*80)
    print()
    print(f"Total experiments: {len(df)}")
    print(f"Encoders: {df['encoder'].unique().tolist()}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")
    print()
    print("Files saved:")
    print(f"  - {output_path}")
    print(f"  - {summary_path}")

if __name__ == "__main__":
    main()
