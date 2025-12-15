#!/usr/bin/env python3
"""
aggregate_layerwise_results.py

Aggregate layer-wise MI results across multiple seeds and compute statistics.

Usage:
    python aggregate_layerwise_results.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RESULTS_DIR = HOME / "results" / "layerwise_mi_multi"
OUTPUT_DIR = HOME / "results" / "layerwise_aggregated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ['bert', 'roberta', 'sroberta']
ENCODER_NAMES = {
    'bert': 'BERT-base',
    'roberta': 'RoBERTa-base',
    'sroberta': 'Sentence-RoBERTa'
}
TASKS = ['4way', '6way']
SPLITS = ['train', 'val', 'test']
SEEDS = [42, 43, 44, 45, 46]


def load_results(encoder, task, split):
    """Load results for all seeds"""
    results = []
    
    for seed in SEEDS:
        csv_file = RESULTS_DIR / f"layerwise_mi_{encoder}_{task}_{split}_seed{seed}.csv"
        json_file = RESULTS_DIR / f"summary_{encoder}_{task}_{split}_seed{seed}.json"
        
        if csv_file.exists() and json_file.exists():
            df = pd.read_csv(csv_file)
            with open(json_file) as f:
                summary = json.load(f)
            
            results.append({
                'seed': seed,
                'df': df,
                'summary': summary
            })
        else:
            print(f"  ⚠️  Missing: {encoder} {task} {split} seed{seed}")
    
    return results


def aggregate_results(results):
    """Compute mean and std across seeds"""
    if not results:
        return None, None
    
    # Aggregate layer-wise data
    all_dfs = [r['df'] for r in results]
    
    # Compute mean and std for each metric
    metrics = ['mi_encoder_only', 'mi_encoder_sentic', 'delta_mi', 'relative_gain_pct']
    
    aggregated = pd.DataFrame({'layer': all_dfs[0]['layer']})
    
    for metric in metrics:
        values = np.vstack([df[metric].values for df in all_dfs])
        aggregated[f'{metric}_mean'] = values.mean(axis=0)
        aggregated[f'{metric}_std'] = values.std(axis=0)
        aggregated[f'{metric}_min'] = values.min(axis=0)
        aggregated[f'{metric}_max'] = values.max(axis=0)
    
    # Aggregate summary statistics
    summary_agg = {}
    
    # Average gain at layers 9-11
    gains = [r['summary']['relative_gain_pct'] for r in results]
    summary_agg['relative_gain_pct_mean'] = np.mean(gains)
    summary_agg['relative_gain_pct_std'] = np.std(gains)
    summary_agg['relative_gain_pct_min'] = np.min(gains)
    summary_agg['relative_gain_pct_max'] = np.max(gains)
    summary_agg['n_seeds'] = len(results)
    summary_agg['seeds'] = SEEDS[:len(results)]
    
    # Per-layer gains at 9, 10, 11
    for layer_idx in [9, 10, 11]:
        layer_gains = [r['summary'][f'delta_mi_layer_{layer_idx}'] for r in results]
        summary_agg[f'delta_mi_layer_{layer_idx}_mean'] = np.mean(layer_gains)
        summary_agg[f'delta_mi_layer_{layer_idx}_std'] = np.std(layer_gains)
    
    return aggregated, summary_agg


def plot_aggregated_results(encoder, task, split, df_agg, summary_agg):
    """Create visualization with error bars"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = df_agg['layer'].values
    
    # Plot 1: MI with error bars
    ax1.errorbar(
        layers, 
        df_agg['mi_encoder_only_mean'], 
        yerr=df_agg['mi_encoder_only_std'],
        label=f'{ENCODER_NAMES[encoder]} only',
        fmt='o-', capsize=3, linewidth=2, markersize=6, color='steelblue'
    )
    ax1.errorbar(
        layers, 
        df_agg['mi_encoder_sentic_mean'], 
        yerr=df_agg['mi_encoder_sentic_std'],
        label=f'{ENCODER_NAMES[encoder]} + SenticNet',
        fmt='s-', capsize=3, linewidth=2, markersize=6, color='coral'
    )
    
    ax1.axvspan(8.5, 11.5, alpha=0.2, color='orange', label='Layers 9-11')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Mutual Information (nats)', fontsize=12)
    ax1.set_title(f'{ENCODER_NAMES[encoder]} | {task.upper()} ({split}) | Layer-wise MI', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 13))
    ax1.set_xticklabels(['Emb'] + [f'{i}' for i in range(1, 13)])
    
    # Plot 2: Relative gain with error bars
    ax2.bar(
        layers, 
        df_agg['relative_gain_pct_mean'], 
        yerr=df_agg['relative_gain_pct_std'],
        color=['orange' if 9 <= l <= 11 else 'steelblue' for l in layers],
        alpha=0.7, capsize=3
    )
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Relative MI Gain (%)', fontsize=12)
    ax2.set_title(f'{ENCODER_NAMES[encoder]} | {task.upper()} ({split}) | SenticNet Contribution', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(0, 13))
    ax2.set_xticklabels(['Emb'] + [f'{i}' for i in range(1, 13)])
    
    # Annotate average gain with std
    avg_gain = summary_agg['relative_gain_pct_mean']
    std_gain = summary_agg['relative_gain_pct_std']
    ax2.annotate(
        f'Avg 9-11: {avg_gain:.1f}% ± {std_gain:.1f}%', 
        xy=(10, avg_gain), 
        xytext=(10, avg_gain + 3 if avg_gain > 0 else avg_gain - 3),
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', color='black')
    )
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"layerwise_mi_{encoder}_{task}_{split}_aggregated.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path.name}")


def create_comparison_table():
    """Create comparison table across encoders"""
    
    print("\n" + "="*70)
    print("COMPARISON TABLE: Average Gain at Layers 9-11")
    print("="*70)
    
    results_table = []
    
    for task in TASKS:
        for split in SPLITS:
            print(f"\n{task.upper()} - {split.upper()}:")
            print("-" * 70)
            
            for encoder in ENCODERS:
                results = load_results(encoder, task, split)
                
                if results:
                    _, summary_agg = aggregate_results(results)
                    
                    mean_gain = summary_agg['relative_gain_pct_mean']
                    std_gain = summary_agg['relative_gain_pct_std']
                    
                    print(f"  {ENCODER_NAMES[encoder]:20s}: {mean_gain:6.2f}% ± {std_gain:5.2f}%")
                    
                    results_table.append({
                        'task': task,
                        'split': split,
                        'encoder': encoder,
                        'encoder_name': ENCODER_NAMES[encoder],
                        'mean_gain': mean_gain,
                        'std_gain': std_gain,
                        'n_seeds': summary_agg['n_seeds']
                    })
                else:
                    print(f"  {ENCODER_NAMES[encoder]:20s}: No data")
    
    # Save comparison table
    df_comparison = pd.DataFrame(results_table)
    df_comparison.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    print(f"\n✓ Saved comparison table to: {OUTPUT_DIR / 'comparison_table.csv'}")
    
    return df_comparison


def create_inverse_correlation_plot(df_comparison):
    """Create plot showing inverse correlation between encoder strength and gain"""
    
    # Focus on train split, 4way task
    df_plot = df_comparison[
        (df_comparison['task'] == '4way') & 
        (df_comparison['split'] == 'train')
    ].copy()
    
    if len(df_plot) < 3:
        print("⚠️  Not enough data for inverse correlation plot")
        return
    
    # Define encoder strength (proxy: model size / training data)
    encoder_strength = {
        'bert': 1,        # BERT-base: 110M params, BookCorpus+Wiki
        'roberta': 2,     # RoBERTa-base: 125M params, 160GB text
        'sroberta': 3     # Sentence-RoBERTa: 125M params + NLI fine-tuning
    }
    
    df_plot['encoder_strength'] = df_plot['encoder'].map(encoder_strength)
    df_plot = df_plot.sort_values('encoder_strength')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(
        df_plot['encoder_strength'],
        df_plot['mean_gain'],
        yerr=df_plot['std_gain'],
        fmt='o-', capsize=5, linewidth=2, markersize=10,
        color='steelblue', ecolor='coral', elinewidth=2
    )
    
    # Annotate points
    for _, row in df_plot.iterrows():
        ax.annotate(
            row['encoder_name'],
            xy=(row['encoder_strength'], row['mean_gain']),
            xytext=(0, 10), textcoords='offset points',
            ha='center', fontsize=10, fontweight='bold'
        )
    
    ax.set_xlabel('Encoder Strength', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lexical Contribution (%)', fontsize=12, fontweight='bold')
    ax.set_title('Inverse Correlation: Encoder Strength vs Lexical Gain\n(4-way, train split)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Weak\n(BERT)', 'Medium\n(RoBERTa)', 'Strong\n(S-RoBERTa)'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "inverse_correlation_encoder_strength.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved inverse correlation plot: {output_path.name}")


def main():
    print("="*70)
    print("AGGREGATING LAYER-WISE MI RESULTS")
    print("="*70)
    print()
    
    # Process all combinations
    for task in TASKS:
        for split in SPLITS:
            print(f"\n{task.upper()} - {split.upper()}")
            print("-" * 70)
            
            for encoder in ENCODERS:
                results = load_results(encoder, task, split)
                
                if not results:
                    print(f"  {ENCODER_NAMES[encoder]:20s}: No data")
                    continue
                
                print(f"  {ENCODER_NAMES[encoder]:20s}: {len(results)} seeds")
                
                # Aggregate
                df_agg, summary_agg = aggregate_results(results)
                
                # Save aggregated data
                csv_out = OUTPUT_DIR / f"layerwise_mi_{encoder}_{task}_{split}_aggregated.csv"
                df_agg.to_csv(csv_out, index=False)
                
                json_out = OUTPUT_DIR / f"summary_{encoder}_{task}_{split}_aggregated.json"
                with open(json_out, 'w') as f:
                    json.dump(summary_agg, f, indent=2)
                
                # Create plot
                plot_aggregated_results(encoder, task, split, df_agg, summary_agg)
                
                print(f"    Mean gain (layers 9-11): {summary_agg['relative_gain_pct_mean']:.2f}% ± {summary_agg['relative_gain_pct_std']:.2f}%")
    
    # Create comparison table
    print()
    df_comparison = create_comparison_table()
    
    # Create inverse correlation plot
    print()
    create_inverse_correlation_plot(df_comparison)
    
    print("\n" + "="*70)
    print("AGGREGATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()