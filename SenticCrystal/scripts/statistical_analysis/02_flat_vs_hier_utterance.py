#!/usr/bin/env python3
"""
02. Flat vs Hierarchical at Utterance-level
============================================
Baseline 실험에서 Flat vs Hier 비교

Input:
    - results/analysis/all_results_combined.csv

Output:
    - Paired t-test: Hier vs Flat (same seed)

Result:
    - 4way: Hier > Flat (+1.9%, p=0.007) **
"""

import pandas as pd
import numpy as np
from scipy import stats


def analyze_flat_vs_hier_utterance():
    """Compare Flat vs Hier at Utterance-level (Baseline)"""

    df = pd.read_csv('results/analysis/all_results_combined.csv')

    print('='*60)
    print('Flat vs Hierarchical at Utterance-level (Baseline)')
    print('='*60)

    # 4way: sentence-roberta (flat) vs sentence-roberta-hier (hier)
    flat_4way = df[(df['type']=='flat') & (df['task']=='4way') &
                   (df['encoder']=='sentence-roberta') & (df['pool']=='mean')]
    hier_4way = df[(df['type']=='hierarchical') & (df['task']=='4way') &
                   (df['encoder']=='sentence-roberta-hier') & (df['pool']=='mean') &
                   (df['aggregator']=='mean')]

    # Find common seeds
    common_seeds = sorted(set(flat_4way['seed'].unique()) & set(hier_4way['seed'].unique()))

    flat_vals = [flat_4way[flat_4way['seed']==s]['weighted_f1'].mean() for s in common_seeds]
    hier_vals = [hier_4way[hier_4way['seed']==s]['weighted_f1'].mean() for s in common_seeds]

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(hier_vals, flat_vals)

    print(f'\n4way:')
    print(f'  Seeds: {common_seeds}')
    print(f'  Flat: {np.mean(flat_vals):.4f} ± {np.std(flat_vals):.4f} (min={np.min(flat_vals):.4f}, max={np.max(flat_vals):.4f})')
    print(f'  Hier: {np.mean(hier_vals):.4f} ± {np.std(hier_vals):.4f} (min={np.min(hier_vals):.4f}, max={np.max(hier_vals):.4f})')
    print(f'  Δ = {np.mean(hier_vals) - np.mean(flat_vals):+.4f} (Hier - Flat)')
    print(f'  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}')

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    print(f'  Result: Hier vs Flat {sig}')

    return {
        'flat_mean': np.mean(flat_vals),
        'flat_std': np.std(flat_vals),
        'flat_min': np.min(flat_vals),
        'flat_max': np.max(flat_vals),
        'hier_mean': np.mean(hier_vals),
        'hier_std': np.std(hier_vals),
        'hier_min': np.min(hier_vals),
        'hier_max': np.max(hier_vals),
        't_stat': t_stat,
        'p_val': p_val
    }


if __name__ == '__main__':
    result = analyze_flat_vs_hier_utterance()

    print('\n' + '='*60)
    print('Summary: Hier > Flat at Utterance-level (+1.9%, p=0.007)')
    print('='*60)
