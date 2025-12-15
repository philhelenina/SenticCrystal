#!/usr/bin/env python3
"""
06. Discourse Markers Position Analysis
========================================
발화 내 담화표지 위치 분석 - 감정별 periphery 분포

Input:
    - results/discourse_markers/markers_4way_extracted.csv
    - results/discourse_markers/markers_6way_extracted.csv

Output:
    - L/R ratio per emotion
    - ANOVA for position differences
    - Pairwise t-tests with Bonferroni correction
    - Right periphery analysis (Haselow theory): though, however, then

Result:
    - Sad: right periphery에 담화표지가 더 많음 (mean_pos=0.453)
    - Happy vs Sad: p=0.0050 (Bonferroni significant)
    - Sad vs Neutral: p=0.0024 (Bonferroni significant)
    - Right periphery markers (though/however): Neutral > Sad (χ²=11.31, p=0.0102)
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


def analyze_discourse_markers(task='4way'):
    """Analyze discourse marker positions by emotion"""

    emotions = {
        '4way': {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'},
        '6way': {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral', 4: 'excited', 5: 'frustrated'}
    }[task]

    df = pd.read_csv(f'results/discourse_markers/markers_{task}_extracted.csv')
    df_labeled = df[df['emotion'] >= 0].copy()

    print(f'\n{task.upper()} Discourse Markers Analysis')
    print('='*60)
    print(f'Total markers: {len(df)}')
    print(f'Labeled markers: {len(df_labeled)}')

    # 1. Position Distribution by Emotion
    print('\n1. Position Distribution by Emotion')
    print('-'*60)
    print(f'{"Emotion":10s} | {"n":>5s} | {"L":>5s} | {"R":>5s} | {"L/R":>5s} | {"mean_pos":>8s}')
    print('-'*60)

    for emo_id, emo_name in emotions.items():
        emo_df = df_labeled[df_labeled['emotion'] == emo_id]
        if len(emo_df) == 0:
            continue

        left = (emo_df['position'] < 0.5).sum()
        right = (emo_df['position'] >= 0.5).sum()
        lr_ratio = left / right if right > 0 else float('inf')
        mean_pos = emo_df['position'].mean()

        print(f'{emo_name.upper():10s} | {len(emo_df):>5d} | {left:>5d} | {right:>5d} | {lr_ratio:>5.2f} | {mean_pos:>8.3f}')

    # Overall
    left_all = (df_labeled['position'] < 0.5).sum()
    right_all = (df_labeled['position'] >= 0.5).sum()
    lr_all = left_all / right_all if right_all > 0 else float('inf')
    print('-'*60)
    print(f'{"OVERALL":10s} | {len(df_labeled):>5d} | {left_all:>5d} | {right_all:>5d} | {lr_all:>5.2f} | {df_labeled["position"].mean():>8.3f}')

    # 2. Statistical Tests
    print('\n2. Statistical Tests')
    print('-'*60)

    # ANOVA
    groups = [df_labeled[df_labeled['emotion']==i]['position'].values
              for i in emotions.keys()
              if len(df_labeled[df_labeled['emotion']==i]) > 0]
    f_stat, p_val = stats.f_oneway(*groups)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    print(f'One-way ANOVA: F={f_stat:.3f}, p={p_val:.4f} {sig}')

    # Pairwise t-tests with Bonferroni
    valid_emotions = [e for e in emotions.keys() if len(df_labeled[df_labeled['emotion']==e]) > 0]
    n_comparisons = len(list(combinations(valid_emotions, 2)))
    alpha_bonf = 0.05 / n_comparisons

    print(f'\nPairwise t-tests (Bonferroni α = {alpha_bonf:.4f}):')
    print('-'*60)

    significant_pairs = []
    for i, j in combinations(valid_emotions, 2):
        pos_i = df_labeled[df_labeled['emotion']==i]['position']
        pos_j = df_labeled[df_labeled['emotion']==j]['position']
        t, p = stats.ttest_ind(pos_i, pos_j)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        bonf = '✓' if p < alpha_bonf else ''
        print(f'  {emotions[i]:8s} vs {emotions[j]:8s}: t={t:>6.2f}, p={p:.4f} {sig} {bonf}')

        if p < alpha_bonf:
            significant_pairs.append((emotions[i], emotions[j], p))

    return {
        'n_labeled': len(df_labeled),
        'anova_f': f_stat,
        'anova_p': p_val,
        'significant_pairs': significant_pairs,
        'df': df_labeled
    }


def analyze_right_periphery(df, emotions):
    """
    Analyze right periphery markers (Haselow theory)
    Focus on: though, however, then - markers of subjectivity at utterance end
    """
    print('\n3. Right Periphery Analysis (Haselow Theory)')
    print('-'*60)
    print('Markers: though, however (subjectivity/stance markers)')
    print('-'*60)

    # Filter for though/however markers
    rp_markers = df[df['marker'].str.lower().isin(['though', 'however'])]

    if len(rp_markers) == 0:
        print('No though/however markers found')
        return None

    print(f'Total though/however markers: {len(rp_markers)}')
    print(f'\n{"Emotion":10s} | {"n":>4s} | {"mean_pos":>8s} | {"at_end(>0.7)":>12s}')
    print('-'*60)

    # Contingency table for chi-square
    contingency = []

    for emo_id, emo_name in emotions.items():
        emo_df = rp_markers[rp_markers['emotion'] == emo_id]
        if len(emo_df) == 0:
            continue

        mean_pos = emo_df['position'].mean()
        at_end = (emo_df['position'] > 0.7).sum()
        not_at_end = len(emo_df) - at_end

        contingency.append([at_end, not_at_end])

        pct = at_end / len(emo_df) * 100
        print(f'{emo_name.upper():10s} | {len(emo_df):>4d} | {mean_pos:>8.3f} | {at_end:>3d}/{len(emo_df):<3d} ({pct:>5.1f}%)')

    # Chi-square test
    if len(contingency) >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'\nChi-square test: χ²={chi2:.3f}, p={p:.4f} {sig}')
        print(f'(Testing if at_end rate differs by emotion)')

    return {'chi2': chi2, 'p': p}


def analyze_marker_position_by_emotion(df, emotions, left_thresh=0.2, right_thresh=0.8):
    """
    Analyze specific marker positions by emotion using Chi-square test
    Test whether marker position (left/mid/right) differs by emotion
    """
    print('\n4. Marker-Specific Position Analysis (Chi-Square)')
    print('-'*70)
    print(f'LEFT: position < {left_thresh}, RIGHT: position > {right_thresh}')
    print('-'*70)

    df['marker_lower'] = df['marker'].str.lower()

    top_markers = ['maybe', 'though', 'and', 'so', 'like', 'well', 'but', 'oh', 'you know', 'i mean']

    significant_markers = []

    for marker in top_markers:
        marker_df = df[(df['marker_lower'] == marker) & (df['emotion'] >= 0)]

        if len(marker_df) < 30:
            continue

        # Build contingency table
        contingency = []
        emo_names = []

        for emo_id, emo_name in emotions.items():
            emo_marker = marker_df[marker_df['emotion'] == emo_id]
            if len(emo_marker) < 5:
                continue

            left = (emo_marker['position'] < left_thresh).sum()
            right = (emo_marker['position'] > right_thresh).sum()
            mid = len(emo_marker) - left - right

            contingency.append([left, mid, right])
            emo_names.append(emo_name)

        if len(contingency) < 2:
            continue

        import numpy as np
        contingency = np.array(contingency)

        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        if p < 0.05:
            print(f'\n{marker.upper():15s} (n={len(marker_df)}) χ²={chi2:.2f}, p={p:.4f} {sig}')
            for i, name in enumerate(emo_names):
                total = contingency[i].sum()
                l_pct = contingency[i][0] / total * 100
                r_pct = contingency[i][2] / total * 100
                print(f'  {name:8s}: L={l_pct:5.1f}%, R={r_pct:5.1f}%')
            significant_markers.append((marker, chi2, p))

    return significant_markers


def analyze_lp_rp_by_emotion(df, emotions, lp_thresh=0.15, rp_thresh=0.85):
    """
    Analyze Left Periphery / Right Periphery usage by emotion (Haselow Framework)

    LP: discourse coherence, turn-taking, topic shifts
    RP: illocutionary modification, stance marking, turn-giving
    """
    print('\n5. LP/RP Functional Analysis (Haselow Framework)')
    print('-'*70)
    print(f'LP: position < {lp_thresh} | RP: position > {rp_thresh}')
    print('-'*70)

    df['marker_lower'] = df['marker'].str.lower()

    markers = ['well', 'oh', 'but', 'so', 'and', 'like', 'you know', 'i mean',
               'maybe', 'though', 'i guess', 'i think', 'probably']

    significant_results = []

    for marker in markers:
        marker_df = df[(df['marker_lower'] == marker) & (df['emotion'] >= 0)]
        if len(marker_df) < 30:
            continue

        contingency = []
        emo_names = []
        emo_data = []

        for emo_id, emo_name in emotions.items():
            emo_marker = marker_df[marker_df['emotion'] == emo_id]
            if len(emo_marker) < 5:
                continue

            lp = (emo_marker['position'] < lp_thresh).sum()
            rp = (emo_marker['position'] > rp_thresh).sum()
            mid = len(emo_marker) - lp - rp
            total = len(emo_marker)

            contingency.append([lp, mid, rp])
            emo_names.append(emo_name)
            emo_data.append({'name': emo_name, 'total': total,
                           'lp_pct': lp/total*100, 'rp_pct': rp/total*100})

        if len(contingency) < 2:
            continue

        import numpy as np
        contingency = np.array(contingency)
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        if p < 0.05:
            print(f'\n{marker.upper():12s} (n={len(marker_df)}) χ²={chi2:.2f}, p={p:.4f} {sig}')
            for d in emo_data:
                print(f'  {d["name"].upper():8s}: LP={d["lp_pct"]:5.1f}%, RP={d["rp_pct"]:5.1f}%')
            significant_results.append((marker, chi2, p, emo_data))

    return significant_results


if __name__ == '__main__':
    print('#'*60)
    print('# Discourse Markers Position Analysis')
    print('#'*60)

    emotions_4way = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}

    for task in ['4way', '6way']:
        try:
            result = analyze_discourse_markers(task)
            # Additional analyses only for 4way
            if task == '4way' and result.get('df') is not None:
                analyze_right_periphery(result['df'], emotions_4way)
                analyze_marker_position_by_emotion(result['df'], emotions_4way)
                analyze_lp_rp_by_emotion(result['df'], emotions_4way)
        except FileNotFoundError:
            print(f'\n{task}: File not found')

    print('\n' + '='*70)
    print('Summary (Haselow LP/RP Framework):')
    print('='*70)
    print('1. Overall Position (ANOVA p=0.0101):')
    print('   - Sad: mean_pos=0.453 (markers toward RP)')
    print('   - Happy vs Sad: p=0.0050, Sad vs Neutral: p=0.0024')
    print('')
    print('2. LP-Dominant Markers (Turn-taking):')
    print('   - WELL: 83% LP (turn-taking), all emotions')
    print('   - OH: 81% LP (recognition/topic shift)')
    print('')
    print('3. Emotion-Specific LP/RP Patterns (6 sig. markers):')
    print('   - AND: SAD 21% LP (continuation), ANGRY 19% RP (afterthought)')
    print('   - MAYBE: SAD 75% LP (hedge upfront), ANGRY 33% RP (soften after)')
    print('   - SO: SAD 11% LP (delays conclusion), NEUTRAL 21% LP')
    print('   - THOUGH: NEUTRAL 45% RP (stance mod), ANGRY 0% RP')
    print('   - WELL: SAD 5% RP (mitigation), others ~0% RP')
    print('   - LIKE: ANGRY 21% LP (filler), SAD 11% LP')
    print('')
    print('4. Emotion Profiles:')
    print('   - ANGRY: Assertive + afterthought (MAYBE/AND at RP), no stance mod')
    print('   - SAD: Hedging upfront (MAYBE LP), continuation (AND LP), delayed conclusion')
    print('   - HAPPY: Balanced LP/RP usage, flexible')
    print('   - NEUTRAL: Formal turn-taking (WELL LP), stance mod (THOUGH RP)')
    print('='*70)
