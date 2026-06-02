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
    speakers = df['Speaker'].unique()
    print(f"Normalizing F0 for {len(speakers)} speakers...")

    # Store speaker mean F0 for reference
    speaker_mean_f0 = {}

    for speaker in speakers:
        speaker_mask = df['Speaker'] == speaker
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
            speaker_mask = df['Speaker'] == speaker

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
    adds a new column for each z-score normalized feature.
    """
    df = df.copy()
    speakers = df['Speaker'].unique()

    for col in columns:
        if col not in df.columns:
            continue

        new_col = col + '_speakerZ'
        df[new_col] = np.nan

        for speaker in speakers:
            speaker_mask = df['Speaker'] == speaker
            speaker_vals = df.loc[speaker_mask, col]

            mean_val = speaker_vals.mean()
            std_val = speaker_vals.std()

            if std_val > 0:
                df.loc[speaker_mask, new_col] = (speaker_vals - mean_val) / std_val
            else:
                df.loc[speaker_mask, new_col] = 0

    return df

def remove_speaker_outliers(df, column, sd_threshold=3):
    """
    Removes rows where the specified column value is more than 
    SD_threshold standard deviations from the speaker's mean.
    """
    df_clean = pd.DataFrame()
    speakers = df['Speaker'].unique()
    
    initial_count = len(df)
    
    for speaker in speakers:
        speaker_mask = df['Speaker'] == speaker
        speaker_data = df[speaker_mask].copy()
        
        mean = speaker_data[column].mean()
        std = speaker_data[column].std()
        
        lower_bound = mean - (sd_threshold * std)
        upper_bound = mean + (sd_threshold * std)
        
        filtered_data = speaker_data[
            (speaker_data[column] >= lower_bound) & 
            (speaker_data[column] <= upper_bound)
        ]
        df_clean = pd.concat([df_clean, filtered_data])
        
    print(f"Outlier Removal: Dropped {initial_count - len(df_clean)} rows based on {column}")
    return df_clean

def create_llm_acoustic_prompts(df):
    """
    Computes true standard deviation metrics from the normalized data
    and generates a clean text block column for LLM injection.
    """
    df = df.copy()
    prompts = []
    
    # calculate speaker-level standard deviations for the F0 semitone column
    f0_norm_col = 'F0semitone_speakerNorm_sma3nz_amean'
    f0_sds = df.groupby('Speaker')[f0_norm_col].transform('std')
    
    for idx, row in df.iterrows():
        speaker = row['Speaker']
        
        # 1. F0 Mean Z-score - the distance from mean divided by the speaker pitch standard deviation
        f0_sd = f0_sds[idx]
        f0_mean_z = row[f0_norm_col] / f0_sd if (pd.notna(f0_sd) and f0_sd > 0) else 0
        
        # 2. F0 Range Z-score (Spread of pitch)
        f0_std_col = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
        # Compare current utterance spread to speaker's average spread
        speaker_rows = df[df['Speaker'] == speaker]
        range_mean = speaker_rows[f0_std_col].mean()
        range_std = speaker_rows[f0_std_col].std()
        f0_range_z = (row[f0_std_col] - range_mean) / range_std if (pd.notna(range_std) and range_std > 0) else 0

        # 3. Intensity Z-score (Loudness)
        loud_col = 'loudness_sma3_amean_speakerZ'
        intensity_z = row[loud_col] if loud_col in df.columns else 0

        # 4. Speech Rate (% change vs speaker average)
        rate_col = 'VoicedSegmentsPerSec'
        rate_baseline = speaker_rows[rate_col].mean()
        speech_rate_pct = ((row[rate_col] - rate_baseline) / rate_baseline) * 100 if (pd.notna(rate_baseline) and rate_baseline > 0) else 0

        # 5. Voice Quality (Jitter Z-score)
        jitter_col = 'jitterLocal_sma3nz_amean_speakerZ'
        jitter_z = row[jitter_col] if jitter_col in df.columns else 0
        
        # Qualifiers / Labels for the LLM
        f0_label = "HIGH" if f0_mean_z > 1.2 else "LOW" if f0_mean_z < -1.2 else "NORMAL"
        range_label = "WIDE — emotional activation" if f0_range_z > 1.2 else "MONOTONE" if f0_range_z < -1.2 else "NORMAL"
        loud_label = "LOUD" if intensity_z > 1.2 else "QUIET" if intensity_z < -1.2 else "NORMAL"
        rate_label = "FAST" if speech_rate_pct > 25 else "SLOW" if speech_rate_pct < -25 else "NORMAL"
        voice_quality = "tense" if jitter_z > 1.5 else "shaky/unstable" if jitter_z > 1.0 else "normal"

        # Build the exact string context block
        prompt_block = (
            f"Acoustic deviation summary for {speaker} (compared to their typical baseline):\n"
            f"- F0 mean: {f0_mean_z:+.1f}σ ({f0_label})\n"
            f"- F0 range: {f0_range_z:+.1f}σ ({range_label})\n"
            f"- Intensity: {intensity_z:+.1f}σ ({loud_label})\n"
            f"- Speech rate: {speech_rate_pct:+.0f}% ({rate_label})\n"
            f"- Voice quality: {voice_quality} (jitter {jitter_z:+.1f}σ)"
        )
        prompts.append(prompt_block)
        
    df['acoustic_prompt_injection'] = prompts
    return df

def main():
    # Paths
    base_dir = Path('/home/liaojd/SenticCrystal/scripts/MELD/opensmile')
    features_path = base_dir / 'meld_egemaps_raw.csv'

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

    # winnowing out 3 standard deviation outliers
    df = remove_speaker_outliers(df, 'F0semitoneFrom27.5Hz_sma3nz_amean')
    df = remove_speaker_outliers(df, 'loudness_sma3_amean') 

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

    df = create_llm_acoustic_prompts(df) 

    # Save normalized features
    output_path = base_dir / 'features' / 'egemaps_features_speaker_normalized.csv'
    df.to_csv(output_path, index=False)
    print(f"\nNormalized features saved to: {output_path}")

    print("\n--- SAMPLE INJECTION PROMPT FOR LLM ---")
    for i in range(5):
        print(df['acoustic_prompt_injection'].iloc[i])
    

    # # Compare before/after normalization
    # print("\n" + "="*60)
    # print("Comparison: Before vs After Normalization")
    # print("="*60)

    # test_speakers = ['Ross', 'Phoebe', "Joey", "Rachel", "Chandler"]
    # comparison_df = df[df['Speaker'].isin(test_speakers)]

    # if not comparison_df.empty:
    #     f0_orig = 'F0semitoneFrom27.5Hz_sma3nz_amean'
    #     f0_norm = f0_orig.replace('From27.5Hz', '_speakerNorm')

    #     print("Average Pitch (Mean) by Speaker:")
    #     print("-" * 30)
        

    #     print("BEFORE (27.5Hz Ref):")
    #     print(comparison_df.groupby('Speaker')[f0_orig].mean())
        
    #     print("\nAFTER (Speaker-Specific Ref):")
    #     print(comparison_df.groupby('Speaker')[f0_norm].mean())
        
    #     print("-" * 30)
    #     print("Note: In the 'AFTER' section, both should be extremely close to 0.00,")
    #     print("indicating that the baseline pitch difference has been removed.")
    # else:
    #     print(f"Speakers {test_speakers} not found in the current dataset slice.")

    # --- 7. Statistical Tests (Emotion Analysis) ---
    # print("\n" + "="*60)
    # print("STATISTICAL TESTS: JOY vs. SADNESS")
    # print("="*60)

    # from scipy import stats

    # emotion_a = 'joy'
    # emotion_b = 'sadness'

  
    # df_a = df[df['Emotion'] == emotion_a]
    # df_b = df[df['Emotion'] == emotion_b]

    # if not df_a.empty and not df_b.empty:
    #     # Test the Normalized Pitch
    #     f0_norm_col = 'F0semitone_speakerNorm' 
        
    #     if f0_norm_col in df.columns:
    #         group_a = df_a[f0_norm_col].dropna()
    #         group_b = df_b[f0_norm_col].dropna()
            
    #         t, p = stats.ttest_ind(group_a, group_b)
            
    #         # Calculate Effect Size (Cohen's d)
    #         pooled_std = np.sqrt(((len(group_a)-1)*group_a.std()**2 + (len(group_b)-1)*group_b.std()**2) / (len(group_a)+len(group_b)-2))
    #         d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0
            
    #         sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
    #         print(f"\nFeature: {f0_norm_col}")
    #         print(f"  {emotion_a.capitalize()}: {group_a.mean():.3f} ± {group_a.std():.3f}")
    #         print(f"  {emotion_b.capitalize()}: {group_b.mean():.3f} ± {group_b.std():.3f}")
    #         print(f"  t-stat: {t:.3f}, p-value: {p:.4f} {sig}")
    #         print(f"  Effect Size (Cohen's d): {d:.3f}")
    # else:
    #     print(f"Could not find enough data for {emotion_a} and {emotion_b} to perform tests.")

    # print("\n--- Speaker Bias Check (Ross vs. Phoebe) ---")

    # f0_orig = 'F0semitoneFrom27.5Hz_sma3nz_amean'
    # f0_norm_col = f0_orig.replace('From27.5Hz', '_speakerNorm')

    # # Verify the column exists before calling it
    # if f0_norm_col in df.columns:
    #     ross_pitch = df[df['Speaker'] == 'Ross'][f0_norm_col].dropna()
    #     phoebe_pitch = df[df['Speaker'] == 'Phoebe'][f0_norm_col].dropna()
        
    #     t_stat, p_val = stats.ttest_ind(ross_pitch, phoebe_pitch)
    #     print(f"Post-Normalization Pitch Difference: t={t_stat:.3f}, p={p_val:.4f}")
    # else:
    #     print(f"Error: Could not find column {f0_norm_col}. Check column names in df.columns.")

    return df


if __name__ == '__main__':
    main()
