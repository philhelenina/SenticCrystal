#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_all_results_v2.py
Aggregate BOTH flat baseline and hierarchical results

REAL STRUCTURES:
- Flat: results/baseline/[task]/[encoder]/[layer]/[pool]/[model]/seed_[X]/results.json
- Hier: results/hier_baseline/[task]/[aggregator]_[classifier]/seed_[X]/results.json
        (encoder/layer/pool info is INSIDE results.json)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RESULTS_BASE = HOME / "results"

def collect_flat_results(task="4way"):
    """Collect flat baseline results"""
    
    results = []
    task_dir = RESULTS_BASE / "baseline" / task
    
    if not task_dir.exists():
        print(f"⚠️  Warning: {task_dir} does not exist")
        return pd.DataFrame()
    
    # Find all results.json files
    result_files = list(task_dir.rglob("results.json"))
    
    print(f"  Found {len(result_files)} flat result files for {task}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse path: results/baseline/4way/encoder/layer/pool/model/seed_XX/results.json
            parts = result_file.parts
            
            try:
                task_idx = parts.index(task)
                encoder = parts[task_idx + 1]
                layer = parts[task_idx + 2]
                pool = parts[task_idx + 3]
                model = parts[task_idx + 4]
                seed_str = parts[task_idx + 5]
                seed = int(seed_str.split('_')[1]) if 'seed_' in seed_str else 42
            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Could not parse path: {result_file}")
                continue
            
            # Extract metrics
            metrics = data.get('metrics', {})
            
            result = {
                'type': 'flat',
                'task': task,
                'encoder': encoder,
                'layer': layer,
                'pool': pool,
                'aggregator': None,
                'classifier': model,
                'seed': seed,
                'accuracy': metrics.get('accuracy', np.nan),
                'macro_f1': metrics.get('macro_f1', np.nan),
                'weighted_f1': metrics.get('weighted_f1', np.nan),
                'path': str(result_file)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ⚠️  Error reading {result_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def collect_hierarchical_results(task="4way"):
    """Collect hierarchical results"""
    
    results = []
    task_dir = RESULTS_BASE / "hier_baseline" / task
    
    if not task_dir.exists():
        print(f"⚠️  Warning: {task_dir} does not exist")
        return pd.DataFrame()
    
    # Find all results.json files
    result_files = list(task_dir.rglob("results.json"))
    
    print(f"  Found {len(result_files)} hierarchical result files for {task}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse path: results/hier_baseline/4way/[aggregator]_[classifier]/seed_XX/results.json
            parts = result_file.parts
            
            try:
                task_idx = parts.index(task)
                agg_clf_str = parts[task_idx + 1]  # e.g., "attn_lstm"
                seed_str = parts[task_idx + 2]     # e.g., "seed_42"
                
                # Split aggregator_classifier
                agg_clf_parts = agg_clf_str.split('_')
                if len(agg_clf_parts) >= 2:
                    # Handle cases like "attn_lstm" or "expdecay_mlp"
                    aggregator = '_'.join(agg_clf_parts[:-1])  # everything except last
                    classifier = agg_clf_parts[-1]             # last part
                else:
                    print(f"  ⚠️  Could not parse aggregator_classifier: {agg_clf_str}")
                    continue
                
                seed = int(seed_str.split('_')[1]) if 'seed_' in seed_str else 42
                
            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Could not parse path: {result_file}")
                continue
            
            # Extract info from JSON (encoder, layer, pool are in the JSON!)
            encoder = data.get('embedding_type', 'unknown')
            layer = data.get('layer', 'unknown')
            pool = data.get('pool', 'unknown')
            
            # Also get aggregator and classifier from JSON to verify
            json_agg = data.get('aggregator', aggregator)
            json_clf = data.get('classifier', classifier)
            
            # Use JSON values if they exist (more reliable)
            aggregator = json_agg
            classifier = json_clf
            
            # Extract metrics
            metrics = data.get('metrics', {})
            
            result = {
                'type': 'hierarchical',
                'task': task,
                'encoder': encoder,
                'layer': layer,
                'pool': pool,
                'aggregator': aggregator,
                'classifier': classifier,
                'seed': seed,
                'accuracy': metrics.get('accuracy', np.nan),
                'macro_f1': metrics.get('macro_f1', np.nan),
                'weighted_f1': metrics.get('weighted_f1', np.nan),
                'path': str(result_file)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ⚠️  Error reading {result_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def compute_summary_statistics(df, group_type='flat'):
    """Compute mean and std across seeds for each configuration"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by configuration (exclude seed)
    if group_type == 'flat':
        group_cols = ['type', 'task', 'encoder', 'layer', 'pool', 'classifier']
    else:  # hierarchical
        group_cols = ['type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier']
    
    summary = df.groupby(group_cols).agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'weighted_f1': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    if group_type == 'flat':
        summary.columns = [
            'type', 'task', 'encoder', 'layer', 'pool', 'classifier',
            'acc_mean', 'acc_std', 'n_seeds',
            'macro_f1_mean', 'macro_f1_std',
            'weighted_f1_mean', 'weighted_f1_std'
        ]
    else:
        summary.columns = [
            'type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier',
            'acc_mean', 'acc_std', 'n_seeds',
            'macro_f1_mean', 'macro_f1_std',
            'weighted_f1_mean', 'weighted_f1_std'
        ]
    
    # Round to 4 decimals
    for col in ['acc_mean', 'acc_std', 'macro_f1_mean', 'macro_f1_std', 
                'weighted_f1_mean', 'weighted_f1_std']:
        summary[col] = summary[col].round(4)
    
    return summary

def format_result(mean, std):
    """Format as mean ± std"""
    return f"{mean:.4f} ± {std:.4f}"

def find_best_configs(summary_df, top_k=10):
    """Find top K configurations by weighted F1"""
    
    if summary_df.empty:
        return pd.DataFrame()
    
    # Sort by weighted_f1_mean descending
    best = summary_df.sort_values('weighted_f1_mean', ascending=False).head(top_k).copy()
    
    # Add rank
    best.insert(0, 'rank', range(1, len(best) + 1))
    
    return best

def print_best_configs(best_df, title):
    """Pretty print best configurations"""
    print(f"\n📊 {title}")
    print(f"{'='*100}")
    
    for idx, row in best_df.iterrows():
        config_str = f"{row['encoder']:25s} | {row['layer']:10s} | {row['pool']:15s}"
        
        if 'aggregator' in row and pd.notna(row['aggregator']):
            config_str += f" | {row['aggregator']:8s}"
        
        config_str += f" | {row['classifier']:4s}"
        
        print(f"{row['rank']:2d}. {config_str} | WF1: {row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f}")

def main():
    print("="*100)
    print("COMPREHENSIVE RESULTS AGGREGATION (Flat + Hierarchical)")
    print("="*100)
    print()
    
    all_results = []
    all_summaries = []
    
    # Process both tasks
    for task in ["4way", "6way"]:
        print(f"\n{'='*100}")
        print(f"Processing {task.upper()}")
        print(f"{'='*100}\n")
        
        # Collect flat baseline
        print("📦 Collecting Flat Baseline...")
        flat_df = collect_flat_results(task)
        
        # Collect hierarchical
        print("📦 Collecting Hierarchical...")
        hier_df = collect_hierarchical_results(task)
        
        # Combine
        task_df = pd.concat([flat_df, hier_df], ignore_index=True)
        
        if task_df.empty:
            print(f"⚠️  No results found for {task}")
            continue
        
        print(f"\n✅ Total collected: {len(task_df)} results")
        print(f"   Flat: {len(flat_df)}")
        print(f"   Hierarchical: {len(hier_df)}")
        print(f"   Seeds: {task_df['seed'].nunique()}")
        
        all_results.append(task_df)
        
        # Compute summaries separately for flat and hier
        print(f"\n📊 Computing summary statistics...")
        
        flat_summary = compute_summary_statistics(flat_df[flat_df['type']=='flat'], 'flat') if not flat_df.empty else pd.DataFrame()
        hier_summary = compute_summary_statistics(hier_df[hier_df['type']=='hierarchical'], 'hierarchical') if not hier_df.empty else pd.DataFrame()
        
        # Combine summaries
        if not flat_summary.empty and not hier_summary.empty:
            if 'aggregator' not in flat_summary.columns:
                flat_summary['aggregator'] = None
            col_order = ['type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier',
                        'acc_mean', 'acc_std', 'n_seeds', 'macro_f1_mean', 'macro_f1_std',
                        'weighted_f1_mean', 'weighted_f1_std']
            flat_summary = flat_summary[col_order]
            hier_summary = hier_summary[col_order]
            task_summary = pd.concat([flat_summary, hier_summary], ignore_index=True)
        elif not flat_summary.empty:
            if 'aggregator' not in flat_summary.columns:
                flat_summary['aggregator'] = None
            task_summary = flat_summary
        elif not hier_summary.empty:
            task_summary = hier_summary
        else:
            task_summary = pd.DataFrame()
        
        all_summaries.append(task_summary)
        
        # Find best configs
        print(f"\n🏆 Best Configurations:")
        
        if not flat_summary.empty:
            best_flat = find_best_configs(flat_summary, top_k=5)
            print_best_configs(best_flat, f"Top 5 Flat Baseline ({task.upper()})")
        
        if not hier_summary.empty:
            best_hier = find_best_configs(hier_summary, top_k=5)
            print_best_configs(best_hier, f"Top 5 Hierarchical ({task.upper()})")
        
        # Save task-specific results
        out_dir = RESULTS_BASE / "analysis" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Raw results
        task_df.to_csv(out_dir / "all_results.csv", index=False)
        print(f"\n✅ Saved: {out_dir / 'all_results.csv'}")
        
        # Summary statistics
        if not task_summary.empty:
            task_summary.to_csv(out_dir / "summary_statistics.csv", index=False)
            print(f"✅ Saved: {out_dir / 'summary_statistics.csv'}")
        
        # Best configs
        if not flat_summary.empty:
            best_flat = find_best_configs(flat_summary, top_k=10)
            best_flat.to_csv(out_dir / "best_flat.csv", index=False)
            print(f"✅ Saved: {out_dir / 'best_flat.csv'}")
        
        if not hier_summary.empty:
            best_hier = find_best_configs(hier_summary, top_k=10)
            best_hier.to_csv(out_dir / "best_hierarchical.csv", index=False)
            print(f"✅ Saved: {out_dir / 'best_hierarchical.csv'}")
    
    # Combined analysis
    if all_results:
        print(f"\n{'='*100}")
        print("COMBINED ANALYSIS (ALL EXPERIMENTS)")
        print(f"{'='*100}\n")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        
        # Save combined
        out_dir = RESULTS_BASE / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(out_dir / "all_results_combined.csv", index=False)
        print(f"✅ Saved: {out_dir / 'all_results_combined.csv'}")
        
        combined_summary.to_csv(out_dir / "summary_combined.csv", index=False)
        print(f"✅ Saved: {out_dir / 'summary_combined.csv'}")
        
        # Overall best by type and task
        print(f"\n🏆 OVERALL BEST CONFIGURATIONS:")
        print(f"{'='*100}\n")
        
        for task in ["4way", "6way"]:
            for exp_type in ["flat", "hierarchical"]:
                subset = combined_summary[
                    (combined_summary['task'] == task) & 
                    (combined_summary['type'] == exp_type)
                ]
                
                if not subset.empty:
                    best_row = subset.loc[subset['weighted_f1_mean'].idxmax()]
                    
                    print(f"{task.upper()} - {exp_type.upper()}:")
                    print(f"  Encoder:     {best_row['encoder']}")
                    print(f"  Layer:       {best_row['layer']}")
                    print(f"  Pool:        {best_row['pool']}")
                    
                    if exp_type == 'hierarchical' and pd.notna(best_row['aggregator']):
                        print(f"  Aggregator:  {best_row['aggregator']}")
                    
                    print(f"  Classifier:  {best_row['classifier']}")
                    print(f"  Weighted F1: {best_row['weighted_f1_mean']:.4f} ± {best_row['weighted_f1_std']:.4f}")
                    print(f"  Macro F1:    {best_row['macro_f1_mean']:.4f} ± {best_row['macro_f1_std']:.4f}")
                    print(f"  Accuracy:    {best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}")
                    print()
        
        # Comparison: Flat vs Hierarchical
        print(f"\n📊 FLAT vs HIERARCHICAL COMPARISON:")
        print(f"{'='*100}\n")
        
        type_comparison = combined_summary.groupby(['task', 'type']).agg({
            'weighted_f1_mean': ['mean', 'max'],
            'macro_f1_mean': ['mean', 'max'],
            'acc_mean': ['mean', 'max']
        }).round(4)
        
        print(type_comparison)
        type_comparison.to_csv(out_dir / "flat_vs_hierarchical.csv")
        print(f"\n✅ Saved: {out_dir / 'flat_vs_hierarchical.csv'}")
        
        # Encoder comparison (flat only for fair comparison)
        print(f"\n📊 ENCODER COMPARISON (Flat Baseline):")
        print(f"{'='*100}\n")
        
        flat_only = combined_summary[combined_summary['type'] == 'flat']
        if not flat_only.empty:
            encoder_comparison = flat_only.groupby(['task', 'encoder']).agg({
                'weighted_f1_mean': 'mean',
                'macro_f1_mean': 'mean',
                'acc_mean': 'mean'
            }).round(4)
            
            print(encoder_comparison)
            encoder_comparison.to_csv(out_dir / "encoder_comparison.csv")
            print(f"\n✅ Saved: {out_dir / 'encoder_comparison.csv'}")
        
        # Aggregator comparison (hierarchical only)
        print(f"\n📊 AGGREGATOR COMPARISON (Hierarchical):")
        print(f"{'='*100}\n")
        
        hier_only = combined_summary[combined_summary['type'] == 'hierarchical']
        if not hier_only.empty:
            agg_comparison = hier_only.groupby(['task', 'aggregator']).agg({
                'weighted_f1_mean': 'mean',
                'macro_f1_mean': 'mean',
                'acc_mean': 'mean'
            }).round(4)
            
            print(agg_comparison)
            agg_comparison.to_csv(out_dir / "aggregator_comparison.csv")
            print(f"\n✅ Saved: {out_dir / 'aggregator_comparison.csv'}")
        
        # Classifier comparison (MLP vs LSTM, across both types)
        print(f"\n📊 CLASSIFIER COMPARISON (MLP vs LSTM):")
        print(f"{'='*100}\n")
        
        clf_comparison = combined_summary.groupby(['task', 'type', 'classifier']).agg({
            'weighted_f1_mean': 'mean',
            'macro_f1_mean': 'mean',
            'acc_mean': 'mean'
        }).round(4)
        
        print(clf_comparison)
        clf_comparison.to_csv(out_dir / "classifier_comparison.csv")
        print(f"\n✅ Saved: {out_dir / 'classifier_comparison.csv'}")
    
    print(f"\n{'='*100}")
    print("✅ AGGREGATION COMPLETE")
    print(f"{'='*100}")
    print(f"\nAll results saved to: {RESULTS_BASE / 'analysis'}")
    print(f"\nKey files:")
    print(f"  - all_results_combined.csv: Raw data (all seeds)")
    print(f"  - summary_combined.csv: Mean ± std by configuration")
    print(f"  - flat_vs_hierarchical.csv: Flat vs Hierarchical comparison")
    print(f"  - encoder_comparison.csv: Average by encoder (flat only)")
    print(f"  - aggregator_comparison.csv: Average by aggregator (hier only)")
    print(f"  - classifier_comparison.csv: MLP vs LSTM")
    print(f"  - 4way/best_flat.csv: Top 10 flat for 4way")
    print(f"  - 4way/best_hierarchical.csv: Top 10 hierarchical for 4way")
    print(f"  - 6way/best_flat.csv: Top 10 flat for 6way")
    print(f"  - 6way/best_hierarchical.csv: Top 10 hierarchical for 6way")

if __name__ == "__main__":
    main()