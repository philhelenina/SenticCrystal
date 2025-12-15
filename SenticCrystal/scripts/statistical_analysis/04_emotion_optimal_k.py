#!/usr/bin/env python3
"""
04. Emotion-specific Optimal K Analysis
========================================
감정별 최적 K값 분석 및 감정 간 차이 비교

Input:
    - results/turnlevel_k_sweep_bayesian/4way_sentence-roberta_avg_last4_mean_flat/seed*/
    - results/turnlevel_k_sweep_bayesian/6way_sentence-roberta_avg_last4_mean_flat/seed*/

Output:
    - Per-seed optimal K for each emotion
    - Pairwise paired t-tests with Bonferroni correction

Result:
    - 감정별 optimal K 차이는 통계적으로 유의하지 않음 (Bonferroni 보정 후)
"""

import numpy as np
import glob
from scipy import stats
from sklearn.metrics import f1_score
from itertools import combinations
import pandas as pd
from pathlib import Path


def compute_classwise_for_seed(results_dir, task):
    """Compute per-class F1 for each K value"""
    results_dir = Path(results_dir)

    preds = np.load(results_dir / 'preds_perK.npy')
    labels = np.load(results_dir / 'labels.npy')
    Ks = np.load(results_dir / 'Ks.npy')

    emotions = {
        '4way': ['angry', 'happy', 'sad', 'neutral'],
        '6way': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    }[task]

    results = []
    for k_idx, K in enumerate(Ks):
        probs_K = preds[k_idx]
        ypred = np.argmax(probs_K, axis=1)

        mask = labels >= 0
        f1_per_class = f1_score(ypred[mask], labels[mask], average=None, zero_division=0)

        result = {'K': int(K)}
        for i, emotion in enumerate(emotions):
            if i < len(f1_per_class):
                result[f'{emotion}_f1'] = f1_per_class[i]
        results.append(result)

    return pd.DataFrame(results)


def analyze_emotion_optimal_k(task='4way'):
    """Analyze optimal K for each emotion"""

    emotions = {
        '4way': ['angry', 'happy', 'sad', 'neutral'],
        '6way': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    }[task]

    dirs = sorted(glob.glob(f'results/turnlevel_k_sweep_bayesian/{task}_sentence-roberta_avg_last4_mean_flat/seed*'))

    # Collect optimal K per seed per emotion
    optimal_K = {emo: [] for emo in emotions}

    for d in dirs:
        df = compute_classwise_for_seed(d, task)
        for emo in emotions:
            best_idx = df[f'{emo}_f1'].idxmax()
            best_K = df.loc[best_idx, 'K']
            optimal_K[emo].append(best_K)

    print(f'\n{task}: Optimal K per emotion (Mean ± Std)')
    print('-'*40)
    for emo in emotions:
        vals = optimal_K[emo]
        print(f'  {emo.upper():12s}: {np.mean(vals):5.1f} ± {np.std(vals):4.1f}')

    # Pairwise comparisons
    n_comparisons = len(list(combinations(emotions, 2)))
    alpha_bonferroni = 0.05 / n_comparisons

    print(f'\nPairwise Paired t-tests (Bonferroni α = {alpha_bonferroni:.4f}):')
    print('-'*70)
    print(f'{"Pair":24s} | {"mean(d)":>8s} | {"t":>7s} | {"p":>8s} | uncorr | Bonf')
    print('-'*70)

    significant_pairs = []

    for emo1, emo2 in combinations(emotions, 2):
        diff = np.array(optimal_K[emo1]) - np.array(optimal_K[emo2])
        t_stat, p_val = stats.ttest_rel(optimal_K[emo1], optimal_K[emo2])

        sig = '*' if p_val < 0.05 else ''
        bonf = '✓' if p_val < alpha_bonferroni else ''

        print(f'{emo1:10s} vs {emo2:10s} | {np.mean(diff):>+8.1f} | {t_stat:>7.2f} | {p_val:>8.4f} | {sig:>6} | {bonf}')

        if p_val < alpha_bonferroni:
            significant_pairs.append((emo1, emo2, p_val))

    return optimal_K, significant_pairs


if __name__ == '__main__':
    print('='*70)
    print('Emotion-specific Optimal K Analysis')
    print('='*70)

    for task in ['4way', '6way']:
        optimal_K, sig_pairs = analyze_emotion_optimal_k(task)

    print('\n' + '='*70)
    print('Summary:')
    print('  - 감정별 optimal K 차이는 통계적으로 유의하지 않음 (Bonferroni 보정 후)')
    print('  - Exploratory observation으로만 보고 가능')
    print('='*70)
