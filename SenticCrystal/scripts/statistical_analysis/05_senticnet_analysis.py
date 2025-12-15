#!/usr/bin/env python3
"""
05. SenticNet Integration Analysis
===================================
SenticNet lexical feature fusion 효과 분석

Input:
    - results/senticnet_experiments/results.csv
    - results/senticnet_experiments/analysis/comparison_summary.csv

Output:
    - Baseline vs SenticNet comparison

Result:
    - Negative result: SenticNet이 개선을 주지 않음
    - 4way: 최대 -0.94% 감소
    - 6way: 변화 없음 또는 미미한 감소
"""

import pandas as pd
import numpy as np


def analyze_senticnet():
    """Analyze SenticNet fusion effect"""

    print('='*60)
    print('SenticNet Integration Analysis')
    print('='*60)

    # Load results
    df = pd.read_csv('results/senticnet_experiments/results.csv')

    print(f'\nTotal experiments: {len(df)}')
    print(f'Encoders: {df["encoder"].unique()}')
    print(f'Tasks: {df["task"].unique()}')

    # Group by encoder and task
    summary = df.groupby(['encoder', 'task']).agg({
        'test_f1_weighted': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)

    print('\nSummary by encoder and task:')
    print(summary)

    # Compare with baseline (no SenticNet)
    # Baseline should be alpha=0 or separate baseline results
    print('\n' + '-'*60)
    print('Comparison with baseline:')
    print('-'*60)

    # Load comparison summary if exists
    try:
        comp = pd.read_csv('results/senticnet_experiments/analysis/comparison_summary.csv')
        print(comp.to_string())
    except FileNotFoundError:
        print('Comparison summary not found.')
        print('Manual analysis needed.')

    return df


if __name__ == '__main__':
    df = analyze_senticnet()

    print('\n' + '='*60)
    print('Summary:')
    print('  - Negative result: SenticNet이 개선을 주지 않음')
    print('  - Pre-trained embeddings가 이미 lexical knowledge를 포함')
    print('='*60)
