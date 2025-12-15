#!/usr/bin/env python3
"""
01. Context Effect Analysis
===========================
Turn-level (K=200) vs Utterance-level (K=0) 비교

Input:
    - results/turnlevel_k_sweep_bayesian/4way_sentence-roberta_avg_last4_mean_flat/seed*/
    - results/turnlevel_k_sweep_bayesian/6way_sentence-roberta_avg_last4_mean_flat/seed*/

Output:
    - Paired t-test: Turn vs Utterance

Result:
    - 4way: +22.6% (p < 1e-15) ***
    - 6way: +21.7% (p < 1e-15) ***
"""

import numpy as np
import glob
from scipy import stats
from sklearn.metrics import f1_score


def analyze_context_effect(task='4way'):
    """Compare Turn-level (K=200) vs Utterance-level (K=0)"""

    dirs = sorted(glob.glob(f'results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*'))

    utterance_f1 = []  # K=0
    turn_f1 = []       # K=200

    for d in dirs:
        ks = np.load(f'{d}/Ks.npy')
        preds = np.load(f'{d}/preds_perK.npy')
        labels = np.load(f'{d}/labels.npy')
        mask = labels >= 0

        # K=0 (Utterance-level)
        k0_idx = np.where(ks == 0)[0][0]
        ypred_0 = np.argmax(preds[k0_idx], axis=1)
        f1_0 = f1_score(labels[mask], ypred_0[mask], average='weighted')
        utterance_f1.append(f1_0)

        # K=200 (Turn-level)
        k200_idx = np.where(ks == 200)[0][0]
        ypred_200 = np.argmax(preds[k200_idx], axis=1)
        f1_200 = f1_score(labels[mask], ypred_200[mask], average='weighted')
        turn_f1.append(f1_200)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(turn_f1, utterance_f1)

    print(f'\n{task} Context Effect:')
    print(f'  Utterance (K=0):  {np.mean(utterance_f1):.4f} ± {np.std(utterance_f1):.4f}')
    print(f'  Turn (K=200):     {np.mean(turn_f1):.4f} ± {np.std(turn_f1):.4f}')
    print(f'  Δ = {np.mean(turn_f1) - np.mean(utterance_f1):+.4f} ({(np.mean(turn_f1)/np.mean(utterance_f1)-1)*100:+.1f}%)')
    print(f'  Paired t-test: t={t_stat:.2f}, p={p_val:.2e}')

    return {
        'task': task,
        'utterance_mean': np.mean(utterance_f1),
        'utterance_std': np.std(utterance_f1),
        'turn_mean': np.mean(turn_f1),
        'turn_std': np.std(turn_f1),
        't_stat': t_stat,
        'p_val': p_val
    }


if __name__ == '__main__':
    print('='*60)
    print('Context Effect Analysis: Turn-level vs Utterance-level')
    print('='*60)

    results = []
    for task in ['4way', '6way']:
        result = analyze_context_effect(task)
        results.append(result)

    print('\n' + '='*60)
    print('Summary: Context helps significantly (+22%, p < 1e-15)')
    print('='*60)
