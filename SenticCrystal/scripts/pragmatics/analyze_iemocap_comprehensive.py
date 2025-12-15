#!/usr/bin/env python3
"""
11/13/2025
Comprehensive IEMOCAP Dialogue Statistics Analysis
- Dialogue length distribution (p95 analysis for K-sweep)
- Label distribution (4-way vs 6-way)
- Session-wise analysis
- Emotion-specific patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Define paths
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
OUTPUT_DIR = HOME / "results" / "statistics"

def load_all_splits(task='4way'):
    """Load train/val/test splits"""
    data_dir = HOME / "data" / f"iemocap_{task}_data"
    all_dfs = []
    
    print(f"\n📂 Looking for data in: {data_dir}")
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f'{split}_{task}_unified.csv'
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['split'] = split
            all_dfs.append(df)
            print(f"✅ Loaded {split} ({task}): {len(df)} utterances")
        else:
            print(f"⚠️ Missing: {file_path}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Combined: {len(combined)} total utterances")
        return combined
    else:
        return None

def analyze_dialogue_lengths(df, task='4way'):
    """Analyze dialogue length distribution"""
    
    # Group by dialogue
    dialogue_lengths = df.groupby('file_id')['utterance_num'].max() + 1  # +1 for 0-indexed
    
    stats = {
        'task': task,
        'total_dialogues': len(dialogue_lengths),
        'total_utterances': len(df),
        'mean': float(dialogue_lengths.mean()),
        'median': float(dialogue_lengths.median()),
        'std': float(dialogue_lengths.std()),
        'min': int(dialogue_lengths.min()),
        'max': int(dialogue_lengths.max()),
        'p10': float(dialogue_lengths.quantile(0.10)),
        'p25': float(dialogue_lengths.quantile(0.25)),
        'p50': float(dialogue_lengths.quantile(0.50)),
        'p75': float(dialogue_lengths.quantile(0.75)),
        'p90': float(dialogue_lengths.quantile(0.90)),
        'p95': float(dialogue_lengths.quantile(0.95)),
        'p99': float(dialogue_lengths.quantile(0.99)),
    }
    
    # Coverage at different K values
    stats['coverage'] = {
        'K50': float((dialogue_lengths <= 50).mean()),
        'K70': float((dialogue_lengths <= 70).mean()),
        'K100': float((dialogue_lengths <= 100).mean()),
        'K150': float((dialogue_lengths <= 150).mean()),
        'K200': float((dialogue_lengths <= 200).mean()),
    }
    
    return stats, dialogue_lengths

def analyze_by_split(df, task='4way'):
    """Analyze by train/val/test split"""
    
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            dialogue_lengths = split_df.groupby('file_id')['utterance_num'].max() + 1
            split_stats[split] = {
                'n_dialogues': len(dialogue_lengths),
                'n_utterances': len(split_df),
                'mean_length': float(dialogue_lengths.mean()),
                'p95': float(dialogue_lengths.quantile(0.95)),
            }
    
    return split_stats

def analyze_by_emotion(df, task='4way'):
    """Analyze dialogue length by emotion"""
    
    # Filter valid labels (not -1)
    valid_df = df[df['label_num'] >= 0].copy()
    
    emotion_map = {
        0: 'Angry',
        1: 'Happy', 
        2: 'Sad',
        3: 'Neutral',
        4: 'Excited',
        5: 'Frustrated'
    }
    
    emotion_stats = {}
    
    for label_num in sorted(valid_df['label_num'].unique()):
        emo_df = valid_df[valid_df['label_num'] == label_num]
        dialogue_lengths = emo_df.groupby('file_id')['utterance_num'].max() + 1
        
        emotion_name = emotion_map.get(int(label_num), f'Label{label_num}')
        
        if len(dialogue_lengths) > 0:
            emotion_stats[emotion_name] = {
                'n_utterances': len(emo_df),
                'n_dialogues': len(dialogue_lengths),
                'mean_length': float(dialogue_lengths.mean()),
                'p95': float(dialogue_lengths.quantile(0.95)),
            }
    
    return emotion_stats

def analyze_label_distribution(df, task='4way'):
    """Analyze label distribution"""
    
    label_counts = df['label'].value_counts().to_dict()
    
    # Convert to percentages
    total = len(df)
    label_pcts = {k: v/total*100 for k, v in label_counts.items()}
    
    # Valid labels only
    valid_df = df[df['label_num'] >= 0]
    valid_counts = valid_df['label'].value_counts().to_dict()
    valid_total = len(valid_df)
    valid_pcts = {k: v/valid_total*100 for k, v in valid_counts.items()}
    
    return {
        'all_labels': label_counts,
        'all_percentages': label_pcts,
        'valid_labels': valid_counts,
        'valid_percentages': valid_pcts,
        'minus_one_count': label_counts.get('-1', 0),
        'minus_one_percentage': label_pcts.get('-1', 0),
    }

def plot_comprehensive_analysis(df_4way, df_6way, stats_4way, stats_6way):
    """Create comprehensive visualization"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('IEMOCAP Comprehensive Statistics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Dialogue Length Distribution (4-way)
    ax = axes[0, 0]
    lengths_4 = df_4way.groupby('file_id')['utterance_num'].max() + 1
    ax.hist(lengths_4, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(stats_4way['p95'], color='red', linestyle='--', linewidth=2, label=f"p95={stats_4way['p95']:.0f}")
    ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='K=100')
    ax.set_xlabel('Dialogue Length (turns)')
    ax.set_ylabel('Frequency')
    ax.set_title('4-way: Dialogue Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Dialogue Length Distribution (6-way)
    ax = axes[0, 1]
    lengths_6 = df_6way.groupby('file_id')['utterance_num'].max() + 1
    ax.hist(lengths_6, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(stats_6way['p95'], color='red', linestyle='--', linewidth=2, label=f"p95={stats_6way['p95']:.0f}")
    ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='K=100')
    ax.set_xlabel('Dialogue Length (turns)')
    ax.set_ylabel('Frequency')
    ax.set_title('6-way: Dialogue Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. CDF Comparison
    ax = axes[0, 2]
    sorted_4 = np.sort(lengths_4)
    sorted_6 = np.sort(lengths_6)
    cdf_4 = np.arange(1, len(sorted_4)+1) / len(sorted_4)
    cdf_6 = np.arange(1, len(sorted_6)+1) / len(sorted_6)
    ax.plot(sorted_4, cdf_4, label='4-way', linewidth=2)
    ax.plot(sorted_6, cdf_6, label='6-way', linewidth=2)
    ax.axvline(100, color='orange', linestyle='--', alpha=0.7, label='K=100')
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='p95')
    ax.set_xlabel('Dialogue Length (turns)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF: K Coverage Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Box plot by split (4-way)
    ax = axes[1, 0]
    split_data_4 = []
    split_labels_4 = []
    for split in ['train', 'val', 'test']:
        split_df = df_4way[df_4way['split'] == split]
        if len(split_df) > 0:
            lengths = split_df.groupby('file_id')['utterance_num'].max() + 1
            split_data_4.append(lengths)
            split_labels_4.append(f"{split}\n(n={len(lengths)})")
    ax.boxplot(split_data_4, labels=split_labels_4)
    ax.set_ylabel('Dialogue Length (turns)')
    ax.set_title('4-way: Length by Split')
    ax.grid(True, alpha=0.3)
    
    # 5. Box plot by split (6-way)
    ax = axes[1, 1]
    split_data_6 = []
    split_labels_6 = []
    for split in ['train', 'val', 'test']:
        split_df = df_6way[df_6way['split'] == split]
        if len(split_df) > 0:
            lengths = split_df.groupby('file_id')['utterance_num'].max() + 1
            split_data_6.append(lengths)
            split_labels_6.append(f"{split}\n(n={len(lengths)})")
    ax.boxplot(split_data_6, labels=split_labels_6)
    ax.set_ylabel('Dialogue Length (turns)')
    ax.set_title('6-way: Length by Split')
    ax.grid(True, alpha=0.3)
    
    # 6. Label distribution comparison
    ax = axes[1, 2]
    labels_4 = df_4way[df_4way['label_num'] >= 0]['label'].value_counts()
    labels_6 = df_6way[df_6way['label_num'] >= 0]['label'].value_counts()
    
    x = np.arange(len(labels_4))
    width = 0.35
    ax.bar(x - width/2, labels_4.values, width, label='4-way', color='steelblue')
    
    # Align 6-way labels
    labels_6_aligned = [labels_6.get(label, 0) for label in labels_4.index]
    ax.bar(x + width/2, labels_6_aligned, width, label='6-way', color='coral')
    
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution (Valid Only)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_4.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Emotion-specific length (4-way)
    ax = axes[2, 0]
    emotion_data_4 = []
    emotion_labels_4 = []
    valid_4 = df_4way[df_4way['label_num'] >= 0]
    emotion_map = {0: 'Ang', 1: 'Hap', 2: 'Sad', 3: 'Neu'}
    for label_num in sorted(valid_4['label_num'].unique()):
        emo_df = valid_4[valid_4['label_num'] == label_num]
        if len(emo_df) > 0:
            lengths = emo_df.groupby('file_id')['utterance_num'].max() + 1
            if len(lengths) > 0:
                emotion_data_4.append(lengths)
                emotion_labels_4.append(emotion_map.get(int(label_num), str(label_num)))
    
    if emotion_data_4:
        ax.violinplot(emotion_data_4, positions=range(len(emotion_data_4)), showmeans=True)
        ax.set_xticks(range(len(emotion_labels_4)))
        ax.set_xticklabels(emotion_labels_4)
        ax.set_ylabel('Dialogue Length (turns)')
        ax.set_title('4-way: Length by Emotion')
        ax.grid(True, alpha=0.3)
    
    # 8. Emotion-specific length (6-way)
    ax = axes[2, 1]
    emotion_data_6 = []
    emotion_labels_6 = []
    valid_6 = df_6way[df_6way['label_num'] >= 0]
    emotion_map_6 = {0: 'Ang', 1: 'Hap', 2: 'Sad', 3: 'Neu', 4: 'Exc', 5: 'Fru'}
    for label_num in sorted(valid_6['label_num'].unique()):
        emo_df = valid_6[valid_6['label_num'] == label_num]
        if len(emo_df) > 0:
            lengths = emo_df.groupby('file_id')['utterance_num'].max() + 1
            if len(lengths) > 0:
                emotion_data_6.append(lengths)
                emotion_labels_6.append(emotion_map_6.get(int(label_num), str(label_num)))
    
    if emotion_data_6:
        ax.violinplot(emotion_data_6, positions=range(len(emotion_data_6)), showmeans=True)
        ax.set_xticks(range(len(emotion_labels_6)))
        ax.set_xticklabels(emotion_labels_6, rotation=45)
        ax.set_ylabel('Dialogue Length (turns)')
        ax.set_title('6-way: Length by Emotion')
        ax.grid(True, alpha=0.3)
    
    # 9. Summary statistics table
    ax = axes[2, 2]
    ax.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    ==================
    
    4-WAY:
      Dialogues: {stats_4way['total_dialogues']}
      Utterances: {stats_4way['total_utterances']}
      Mean length: {stats_4way['mean']:.1f}
      Median: {stats_4way['median']:.1f}
      p95: {stats_4way['p95']:.0f} ⭐
      Coverage@K=100: {stats_4way['coverage']['K100']*100:.1f}%
    
    6-WAY:
      Dialogues: {stats_6way['total_dialogues']}
      Utterances: {stats_6way['total_utterances']}
      Mean length: {stats_6way['mean']:.1f}
      Median: {stats_6way['median']:.1f}
      p95: {stats_6way['p95']:.0f} ⭐
      Coverage@K=100: {stats_6way['coverage']['K100']*100:.1f}%
    
    K-SWEEP RECOMMENDATION:
      - p95-based: K=0~{int(max(stats_4way['p95'], stats_6way['p95'])+10)}
      - Conservative: K=0~100 (paper)
      - Exploratory: K=0~{int(max(stats_4way['max'], stats_6way['max']))}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def main():
    print("="*70)
    print("IEMOCAP COMPREHENSIVE STATISTICS ANALYSIS")
    print("="*70)
    print(f"\n🏠 HOME: {HOME}")
    print(f"📁 OUTPUT_DIR: {OUTPUT_DIR}")
    
    # Load data
    print("\n📂 Loading data...")
    df_4way = load_all_splits('4way')
    df_6way = load_all_splits('6way')
    
    if df_4way is None or df_6way is None:
        print("\n❌ Error: Could not load data files")
        print(f"\n📝 Expected data location:")
        print(f"   {HOME}/data/iemocap_4way_data/train_4way_unified.csv")
        print(f"   {HOME}/data/iemocap_4way_data/val_4way_unified.csv")
        print(f"   {HOME}/data/iemocap_4way_data/test_4way_unified.csv")
        print(f"   (same for 6way)")
        return
    
    # Analyze dialogue lengths
    print("\n📊 Analyzing dialogue lengths...")
    stats_4way, lengths_4way = analyze_dialogue_lengths(df_4way, '4way')
    stats_6way, lengths_6way = analyze_dialogue_lengths(df_6way, '6way')
    
    # Print key statistics
    print("\n" + "="*70)
    print("KEY FINDINGS: DIALOGUE LENGTH")
    print("="*70)
    print(f"\n{'Metric':<20} {'4-way':>15} {'6-way':>15}")
    print("-"*50)
    print(f"{'Dialogues':<20} {stats_4way['total_dialogues']:>15} {stats_6way['total_dialogues']:>15}")
    print(f"{'Mean length':<20} {stats_4way['mean']:>15.1f} {stats_6way['mean']:>15.1f}")
    print(f"{'Median':<20} {stats_4way['median']:>15.1f} {stats_6way['median']:>15.1f}")
    print(f"{'p90':<20} {stats_4way['p90']:>15.0f} {stats_6way['p90']:>15.0f}")
    print(f"{'p95 ⭐':<20} {stats_4way['p95']:>15.0f} {stats_6way['p95']:>15.0f}")
    print(f"{'p99':<20} {stats_4way['p99']:>15.0f} {stats_6way['p99']:>15.0f}")
    print(f"{'Max':<20} {stats_4way['max']:>15} {stats_6way['max']:>15}")
    
    print("\n" + "="*70)
    print("COVERAGE AT DIFFERENT K VALUES")
    print("="*70)
    for k in [50, 70, 100, 150, 200]:
        cov_4 = stats_4way['coverage'][f'K{k}'] * 100
        cov_6 = stats_6way['coverage'][f'K{k}'] * 100
        print(f"K={k:<4} {cov_4:>14.1f}% {cov_6:>14.1f}%")
    
    # Analyze by split
    print("\n📋 Analyzing by split...")
    split_stats_4way = analyze_by_split(df_4way, '4way')
    split_stats_6way = analyze_by_split(df_6way, '6way')
    
    # Analyze by emotion
    print("\n😊 Analyzing by emotion...")
    emotion_stats_4way = analyze_by_emotion(df_4way, '4way')
    emotion_stats_6way = analyze_by_emotion(df_6way, '6way')
    
    # Analyze label distribution
    print("\n🏷️ Analyzing label distribution...")
    label_dist_4way = analyze_label_distribution(df_4way, '4way')
    label_dist_6way = analyze_label_distribution(df_6way, '6way')
    
    # Print emotion statistics
    print("\n" + "="*70)
    print("EMOTION-SPECIFIC STATISTICS (4-way)")
    print("="*70)
    for emotion, stats in emotion_stats_4way.items():
        print(f"\n{emotion}:")
        print(f"  Utterances: {stats['n_utterances']}")
        print(f"  Dialogues: {stats['n_dialogues']}")
        print(f"  Mean length: {stats['mean_length']:.1f}")
        print(f"  p95: {stats['p95']:.0f}")
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    fig = plot_comprehensive_analysis(df_4way, df_6way, stats_4way, stats_6way)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(OUTPUT_DIR / 'iemocap_comprehensive_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {OUTPUT_DIR}/iemocap_comprehensive_statistics.png")
    
    # Save JSON report
    report = {
        '4way': {
            'dialogue_stats': stats_4way,
            'split_stats': split_stats_4way,
            'emotion_stats': emotion_stats_4way,
            'label_distribution': label_dist_4way,
        },
        '6way': {
            'dialogue_stats': stats_6way,
            'split_stats': split_stats_6way,
            'emotion_stats': emotion_stats_6way,
            'label_distribution': label_dist_6way,
        },
        'recommendations': {
            'k_sweep_conservative': '0-100 (paper original)',
            'k_sweep_p95_based': f"0-{int(max(stats_4way['p95'], stats_6way['p95'])+10)}",
            'k_sweep_comprehensive': f"0-{int(max(stats_4way['max'], stats_6way['max']))}",
            'k_step_recommendation': 5,
            'reasoning': f"p95={max(stats_4way['p95'], stats_6way['p95']):.0f} suggests K=100 covers {min(stats_4way['coverage']['K100'], stats_6way['coverage']['K100'])*100:.1f}% of dialogues"
        }
    }
    
    with open(OUTPUT_DIR / 'iemocap_statistics_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved report: {OUTPUT_DIR}/iemocap_statistics_report.json")
    
    # Print final recommendation
    print("\n" + "="*70)
    print("🎯 K-SWEEP RANGE RECOMMENDATION")
    print("="*70)
    print(f"\n📌 p95 = {max(stats_4way['p95'], stats_6way['p95']):.0f} turns")
    print(f"   K=100 covers {min(stats_4way['coverage']['K100'], stats_6way['coverage']['K100'])*100:.1f}% of dialogues")
    print(f"\n✅ RECOMMENDED OPTIONS:")
    print(f"   1. Conservative (Paper): K=0~100 (step=5)")
    print(f"   2. Data-driven:          K=0~{int(max(stats_4way['p95'], stats_6way['p95'])+10)} (step=5)")
    print(f"   3. Comprehensive:        K=0~{int(max(stats_4way['max'], stats_6way['max']))} (step=10)")
    print(f"\n💡 Recommendation: Start with Option 1 for paper reproduction,")
    print(f"   then consider Option 2 for additional analysis if time permits.")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
