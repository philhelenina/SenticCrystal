#!/usr/bin/env python3
"""
03. Flat vs Hierarchical at Turn-level
=======================================
Turn-level 실험에서 Flat vs Hier 비교 (모든 K값)

Input:
    - results/turnlevel_k_sweep_bayesian/4way_sentence-roberta_avg_last4_mean_flat/seed*/
    - results/turnlevel_k_sweep_bayesian/4way_sentence-roberta-hier_avg_last4_mean_mean/seed*/
    - results/turnlevel_k_sweep_bayesian/6way_sentence-roberta_avg_last4_mean_flat/seed*/
    - results/turnlevel_k_sweep_bayesian/6way_sentence-roberta-hier_avg_last4_mean_mean/seed*/

Output:
    - Paired t-test at each K value
    - Bonferroni correction (21 comparisons)

Result:
    - 4way K=200: Flat ≈ Hier (+0.4%, p=0.43) n.s.
    - 6way K=170: Flat > Hier (+2.0%, p=0.0009) *** (Bonferroni significant)
"""

import numpy as np
import glob
from scipy import stats
from sklearn.metrics import f1_score


def analyze_flat_vs_hier_turn(task='4way'):
    """Compare Flat vs Hier at Turn-level for all K values"""

    flat_dirs = sorted(glob.glob(f'results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*'))
    hier_dirs = sorted(glob.glob(f'results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta-hier_avg_last4_mean_mean/seed*'))

    # Get K values
    Ks = np.load(f'{flat_dirs[0]}/Ks.npy')
    n_comparisons = len(Ks)
    alpha_bonferroni = 0.05 / n_comparisons

    print(f'\n{task}: Flat vs Hier by K')
    print(f'Bonferroni α = 0.05/{n_comparisons} = {alpha_bonferroni:.4f}')
    print('-'*70)
    print(f'{"K":>4} | {"Flat":>7} | {"Hier":>7} | {"Δ":>7} | {"p-value":>8} | sig | Bonf')
    print('-'*70)

    results = []

    for K in Ks:
        flat_vals = []
        hier_vals = []

        for d in flat_dirs:
            ks = np.load(f'{d}/Ks.npy')
            k_idx = np.where(ks == K)[0]
            if len(k_idx) > 0:
                preds = np.load(f'{d}/preds_perK.npy')
                labels = np.load(f'{d}/labels.npy')
                ypred = np.argmax(preds[k_idx[0]], axis=1)
                mask = labels >= 0
                f1 = f1_score(labels[mask], ypred[mask], average='weighted')
                flat_vals.append(f1)

        for d in hier_dirs:
            ks = np.load(f'{d}/Ks.npy')
            k_idx = np.where(ks == K)[0]
            if len(k_idx) > 0:
                preds = np.load(f'{d}/preds_perK.npy')
                labels = np.load(f'{d}/labels.npy')
                ypred = np.argmax(preds[k_idx[0]], axis=1)
                mask = labels >= 0
                f1 = f1_score(labels[mask], ypred[mask], average='weighted')
                hier_vals.append(f1)

        if len(flat_vals) >= 2 and len(hier_vals) >= 2:
            n = min(len(flat_vals), len(hier_vals))
            t_stat, p_val = stats.ttest_rel(flat_vals[:n], hier_vals[:n])

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            bonf = '✓' if p_val < alpha_bonferroni else ''

            print(f'{int(K):>4} | {np.mean(flat_vals):>7.4f} | {np.mean(hier_vals):>7.4f} | {np.mean(flat_vals)-np.mean(hier_vals):>+7.4f} | {p_val:>8.4f} | {sig:>3} | {bonf}')

            results.append({
                'K': int(K),
                'flat_mean': np.mean(flat_vals),
                'hier_mean': np.mean(hier_vals),
                'delta': np.mean(flat_vals) - np.mean(hier_vals),
                'p_val': p_val,
                'significant': p_val < 0.05,
                'bonferroni_significant': p_val < alpha_bonferroni
            })

    return results


if __name__ == '__main__':
    print('='*70)
    print('Flat vs Hierarchical at Turn-level (all K values)')
    print('='*70)

    for task in ['4way', '6way']:
        results = analyze_flat_vs_hier_turn(task)

    print('\n' + '='*70)
    print('Summary:')
    print('  - Turn-level: Flat ≥ Hier (most K values n.s.)')
    print('  - 6way K=170: Flat > Hier (p=0.0009, Bonferroni significant)')
    print('  - Context가 추가되면 Hierarchical의 이점이 사라짐')
    print('='*70)
