#!/usr/bin/env python3
"""
Speaker-Specific Normalization for Acoustic Features

For F0 features: Convert to semitones using each speaker's mean F0 as reference
Formula: semitone = 12 * log2(f0 / speaker_mean_f0)

For other features: z-score normalization within each speaker

Author: Claude Code
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
from pathlib import Path


def hz_to_semitone(f0_hz, reference_hz):
    """
    Convert F0 in Hz to semitones relative to reference frequency.

    Formula: semitone = 12 * log2(f0 / reference)
    """
    if reference_hz <= 0 or f0_hz <= 0:
        return np.nan
    return 12 * np.log2(f0_hz / reference_hz)


def semitone_to_hz(semitone, reference_hz=27.5):
    """
    Convert semitones (from 27.5Hz reference) back to Hz.

    Formula: f0 = reference * 2^(semitone/12)
    """
    return reference_hz * (2 ** (semitone / 12))


def speaker_normalize_f0(df, f0_columns):
    """
    Normalize F0 features using speaker's mean F0 as reference.

    Steps:
    1. Convert semitones (27.5Hz ref) back to Hz
    2. Calculate speaker's mean F0 in Hz
    3. Re-convert to semitones using speaker's mean as reference
    """
    df = df.copy()

    # Get speaker identifier (file_id contains speaker info)
    speakers = df['file_id'].unique()
    print(f"Normalizing F0 for {len(speakers)} speakers...")

    # Store speaker mean F0 for reference
    speaker_mean_f0 = {}

    for speaker in speakers:
        speaker_mask = df['file_id'] == speaker
        speaker_data = df.loc[speaker_mask]

        # Use the mean F0 column (amean) to get speaker's average pitch
        # First convert from semitones (27.5Hz ref) to Hz
        f0_semitone_col = 'F0semitoneFrom27.5Hz_sma3nz_amean'
        if f0_semitone_col in df.columns:
            speaker_f0_semitones = speaker_data[f0_semitone_col].dropna()
            if len(speaker_f0_semitones) > 0:
                # Convert to Hz
                speaker_f0_hz = semitone_to_hz(speaker_f0_semitones)
                mean_f0_hz = speaker_f0_hz.mean()
                speaker_mean_f0[speaker] = mean_f0_hz

    # Now normalize all F0 columns
    for col in f0_columns:
        if col not in df.columns:
            continue

        new_col = col.replace('From27.5Hz', '_speakerNorm')
        df[new_col] = np.nan

        for speaker in speakers:
            speaker_mask = df['file_id'] == speaker

            if speaker not in speaker_mean_f0:
                continue

            ref_hz = speaker_mean_f0[speaker]

            # Convert original semitones to Hz, then to speaker-normalized semitones
            original_semitones = df.loc[speaker_mask, col]
            f0_hz = semitone_to_hz(original_semitones)
            normalized_semitones = 12 * np.log2(f0_hz / ref_hz)

            df.loc[speaker_mask, new_col] = normalized_semitones

    # Save speaker mean F0 for reference
    speaker_f0_df = pd.DataFrame([
        {'file_id': k, 'mean_f0_hz': v}
        for k, v in speaker_mean_f0.items()
    ])

    return df, speaker_f0_df


def speaker_zscore_normalize(df, columns):
    """
    Z-score normalize features within each speaker.

    Formula: z = (x - speaker_mean) / speaker_std
    """
    df = df.copy()
    speakers = df['file_id'].unique()

    for col in columns:
        if col not in df.columns:
            continue

        new_col = col + '_speakerZ'
        df[new_col] = np.nan

        for speaker in speakers:
            speaker_mask = df['file_id'] == speaker
            speaker_vals = df.loc[speaker_mask, col]

            mean_val = speaker_vals.mean()
            std_val = speaker_vals.std()

            if std_val > 0:
                df.loc[speaker_mask, new_col] = (speaker_vals - mean_val) / std_val
            else:
                df.loc[speaker_mask, new_col] = 0

    return df


def main():
    # Paths
    base_dir = Path('/Users/helenjeong/Projects/DementiaBank-HeaLING/emotion-analysis-test/segments_adress')
    features_path = base_dir / 'features' / 'egemaps_features.csv'

    # Load features
    print("Loading features...")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")

    # Identify F0 columns (semitone from 27.5Hz)
    f0_columns = [col for col in df.columns if 'F0semitoneFrom27.5Hz' in col]
    print(f"\nF0 columns to normalize: {len(f0_columns)}")

    # Identify other acoustic features for z-score normalization
    other_features = [
        'loudness_sma3_amean',
        'jitterLocal_sma3nz_amean',
        'shimmerLocaldB_sma3nz_amean',
        'HNRdBACF_sma3nz_amean',
        'MeanVoicedSegmentLengthSec',
        'VoicedSegmentsPerSec',
    ]

    # Speaker-normalize F0
    print("\n" + "="*60)
    print("Speaker-specific F0 normalization (semitone re-reference)")
    print("="*60)
    df, speaker_f0_df = speaker_normalize_f0(df, f0_columns)

    # Save speaker mean F0
    speaker_f0_path = base_dir / 'features' / 'speaker_mean_f0.csv'
    speaker_f0_df.to_csv(speaker_f0_path, index=False)
    print(f"Speaker mean F0 saved to: {speaker_f0_path}")

    # Z-score normalize other features
    print("\n" + "="*60)
    print("Speaker-specific z-score normalization")
    print("="*60)
    df = speaker_zscore_normalize(df, other_features)

    # Save normalized features
    output_path = base_dir / 'features' / 'egemaps_features_speaker_normalized.csv'
    df.to_csv(output_path, index=False)
    print(f"\nNormalized features saved to: {output_path}")

    # Compare before/after normalization
    print("\n" + "="*60)
    print("Comparison: Before vs After Normalization (PAR only)")
    print("="*60)

    par_df = df[df['speaker'] == 'PAR']

    # F0 comparison
    print("\n--- F0 (mean pitch) ---")
    print("Before (27.5Hz reference):")
    for group in ['control', 'ad']:
        vals = par_df[par_df['group'] == group]['F0semitoneFrom27.5Hz_sma3nz_amean']
        print(f"  {group}: {vals.mean():.2f} ± {vals.std():.2f}")

    print("\nAfter (speaker-specific reference):")
    norm_col = 'F0semitoneFrom27.5Hz_sma3nz_amean'.replace('From27.5Hz', '_speakerNorm')
    if norm_col in par_df.columns:
        for group in ['control', 'ad']:
            vals = par_df[par_df['group'] == group][norm_col].dropna()
            print(f"  {group}: {vals.mean():.2f} ± {vals.std():.2f}")

    # Statistical test after normalization
    from scipy import stats

    print("\n--- Statistical Tests (After Normalization) ---")

    # F0 normalized
    f0_norm_col = 'F0semitoneFrom27.5Hz_sma3nz_amean'.replace('From27.5Hz', '_speakerNorm')
    if f0_norm_col in par_df.columns:
        control = par_df[par_df['group'] == 'control'][f0_norm_col].dropna()
        ad = par_df[par_df['group'] == 'ad'][f0_norm_col].dropna()
        t, p = stats.ttest_ind(control, ad)
        pooled_std = np.sqrt(((len(control)-1)*control.std()**2 + (len(ad)-1)*ad.std()**2) / (len(control)+len(ad)-2))
        d = (control.mean() - ad.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"\nF0 (speaker-normalized):")
        print(f"  Control: {control.mean():.3f} ± {control.std():.3f}")
        print(f"  AD: {ad.mean():.3f} ± {ad.std():.3f}")
        print(f"  t={t:.3f}, p={p:.4f} {sig}, d={d:.3f}")

    # Other z-scored features
    for feat in other_features:
        z_col = feat + '_speakerZ'
        if z_col in par_df.columns:
            control = par_df[par_df['group'] == 'control'][z_col].dropna()
            ad = par_df[par_df['group'] == 'ad'][z_col].dropna()
            t, p = stats.ttest_ind(control, ad)
            pooled_std = np.sqrt(((len(control)-1)*control.std()**2 + (len(ad)-1)*ad.std()**2) / (len(control)+len(ad)-2))
            d = (control.mean() - ad.mean()) / pooled_std if pooled_std > 0 else 0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"\n{feat} (z-scored):")
            print(f"  Control: {control.mean():.3f} ± {control.std():.3f}")
            print(f"  AD: {ad.mean():.3f} ± {ad.std():.3f}")
            print(f"  t={t:.3f}, p={p:.4f} {sig}, d={d:.3f}")

    return df


if __name__ == '__main__':
    main()
