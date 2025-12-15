#!/usr/bin/env python3
"""
11/13/2025
Comprehensive Discourse Marker Statistics Analysis
- Marker frequency by task (4-way vs 6-way)
- Position distribution (L/R ratio)
- Emotion-specific patterns
- Category-wise analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Define paths
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
OUTPUT_DIR = HOME / "results" / "discourse_markers"

# Import discourse marker categories
DISCOURSE_MARKERS = {
    'high_frequency': ['well', 'so', 'but', 'oh', 'and', 'like', 'i mean', 'you know'],
    'epistemic': ['i think', 'i guess', 'i believe', 'maybe', 'probably'],
    'affective': ['unfortunately', 'luckily', 'surprisingly', 'sadly', 'happily'],
    'contrastive': ['but', 'however', 'though', 'although', 'yet'],
    'elaborative': ['and', 'also', 'moreover', 'furthermore'],
    'inferential': ['so', 'therefore', 'thus', 'consequently']
}

def load_data(task='4way'):
    """Load IEMOCAP data"""
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
    return None

def extract_markers(text):
    """Extract discourse markers from text"""
    if pd.isna(text):
        return []
    
    text_lower = str(text).lower()
    tokens = text_lower.split()
    
    found_markers = []
    
    # Check all marker categories
    all_markers = set()
    for markers in DISCOURSE_MARKERS.values():
        all_markers.update(markers)
    
    for marker in all_markers:
        marker_tokens = marker.split()
        
        if len(marker_tokens) == 1:
            # Single token
            for i, token in enumerate(tokens):
                if token == marker or token.startswith(marker):
                    rel_pos = i / max(len(tokens) - 1, 1)
                    found_markers.append({
                        'marker': marker,
                        'position': rel_pos,
                        'absolute_pos': i,
                        'n_tokens': len(tokens)
                    })
        else:
            # Multi-token (e.g., "i mean", "you know")
            for i in range(len(tokens) - len(marker_tokens) + 1):
                if tokens[i:i+len(marker_tokens)] == marker_tokens:
                    rel_pos = i / max(len(tokens) - 1, 1)
                    found_markers.append({
                        'marker': marker,
                        'position': rel_pos,
                        'absolute_pos': i,
                        'n_tokens': len(tokens)
                    })
    
    return found_markers

def analyze_marker_statistics(df, task='4way'):
    """Comprehensive marker statistics"""
    
    all_markers = []
    
    # Extract markers from each utterance
    for idx, row in df.iterrows():
        text = row.get('utterance', row.get('text', ''))
        emotion = int(row.get('label_num', -1)) if not pd.isna(row.get('label_num')) else -1
        
        markers = extract_markers(text)
        
        for m in markers:
            all_markers.append({
                **m,
                'emotion': emotion,
                'task': task,
                'split': row.get('split', 'unknown'),
                'file_id': row.get('file_id', ''),
            })
    
    df_markers = pd.DataFrame(all_markers)
    
    if len(df_markers) == 0:
        print(f"⚠️ No markers found for {task}")
        return None, {}
    
    # Calculate statistics
    stats = {
        'task': task,
        'total_markers': len(df_markers),
        'unique_markers': df_markers['marker'].nunique(),
        'mean_position': float(df_markers['position'].mean()),
        'left_periphery': float((df_markers['position'] < 0.1).mean()),
        'right_periphery': float((df_markers['position'] > 0.9).mean()),
        'lr_ratio': float((df_markers['position'] < 0.1).mean() / max((df_markers['position'] > 0.9).mean(), 0.001)),
    }
    
    # Marker frequencies
    marker_freq = df_markers['marker'].value_counts().to_dict()
    stats['marker_frequencies'] = marker_freq
    stats['top_10_markers'] = dict(list(marker_freq.items())[:10])
    
    # Position stats by marker
    marker_positions = {}
    for marker in df_markers['marker'].unique():
        marker_data = df_markers[df_markers['marker'] == marker]
        marker_positions[marker] = {
            'count': len(marker_data),
            'mean_position': float(marker_data['position'].mean()),
            'left_pct': float((marker_data['position'] < 0.1).mean()),
            'right_pct': float((marker_data['position'] > 0.9).mean()),
            'lr_ratio': float((marker_data['position'] < 0.1).mean() / max((marker_data['position'] > 0.9).mean(), 0.001))
        }
    
    stats['marker_positions'] = marker_positions
    
    # Emotion-specific L/R ratios
    emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral', 4: 'Excited', 5: 'Frustrated'}
    emotion_stats = {}
    
    for emotion in df_markers['emotion'].unique():
        if emotion == -1:
            continue
        
        emo_data = df_markers[df_markers['emotion'] == emotion]
        left = (emo_data['position'] < 0.1).mean()
        right = (emo_data['position'] > 0.9).mean()
        
        emotion_stats[emotion_map.get(emotion, str(emotion))] = {
            'count': len(emo_data),
            'mean_position': float(emo_data['position'].mean()),
            'left_pct': float(left),
            'right_pct': float(right),
            'lr_ratio': float(left / max(right, 0.001))
        }
    
    stats['emotion_stats'] = emotion_stats
    
    return df_markers, stats

def plot_marker_analysis(df_markers_4way, df_markers_6way, stats_4way, stats_6way):
    """Comprehensive visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Discourse Marker Analysis: 4-way vs 6-way', fontsize=14, fontweight='bold')
    
    # 1. Top markers comparison
    ax = axes[0, 0]
    top_4 = pd.Series(stats_4way['top_10_markers']).head(10)
    top_6 = pd.Series(stats_6way['top_10_markers']).head(10)
    
    x = np.arange(len(top_4))
    width = 0.35
    
    ax.bar(x - width/2, top_4.values, width, label='4-way', color='steelblue')
    ax.bar(x + width/2, [top_6.get(m, 0) for m in top_4.index], width, label='6-way', color='coral')
    ax.set_xlabel('Marker')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Markers Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(top_4.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Position distribution (4-way)
    ax = axes[0, 1]
    ax.hist(df_markers_4way['position'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Left periphery')
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='Right periphery')
    ax.set_xlabel('Relative Position (0-1)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'4-way: Position Distribution (L/R={stats_4way["lr_ratio"]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Position distribution (6-way)
    ax = axes[0, 2]
    ax.hist(df_markers_6way['position'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Left periphery')
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='Right periphery')
    ax.set_xlabel('Relative Position (0-1)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'6-way: Position Distribution (L/R={stats_6way["lr_ratio"]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. L/R ratio by emotion (4-way)
    ax = axes[1, 0]
    if stats_4way['emotion_stats']:
        emotions = list(stats_4way['emotion_stats'].keys())
        lr_ratios = [stats_4way['emotion_stats'][e]['lr_ratio'] for e in emotions]
        
        ax.barh(range(len(emotions)), lr_ratios, color='skyblue')
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions)
        ax.set_xlabel('L/R Ratio')
        ax.set_title('4-way: L/R Ratio by Emotion')
        ax.axvline(stats_4way['lr_ratio'], color='red', linestyle='--', label='Overall')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. L/R ratio by emotion (6-way)
    ax = axes[1, 1]
    if stats_6way['emotion_stats']:
        emotions = list(stats_6way['emotion_stats'].keys())
        lr_ratios = [stats_6way['emotion_stats'][e]['lr_ratio'] for e in emotions]
        
        ax.barh(range(len(emotions)), lr_ratios, color='lightcoral')
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions)
        ax.set_xlabel('L/R Ratio')
        ax.set_title('6-way: L/R Ratio by Emotion')
        ax.axvline(stats_6way['lr_ratio'], color='red', linestyle='--', label='Overall')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    SUMMARY STATISTICS
    ==================
    
    4-WAY:
      Total markers: {stats_4way['total_markers']}
      Unique markers: {stats_4way['unique_markers']}
      Mean position: {stats_4way['mean_position']:.3f}
      Left periphery: {stats_4way['left_periphery']:.1%}
      Right periphery: {stats_4way['right_periphery']:.1%}
      L/R ratio: {stats_4way['lr_ratio']:.2f}
    
    6-WAY:
      Total markers: {stats_6way['total_markers']}
      Unique markers: {stats_6way['unique_markers']}
      Mean position: {stats_6way['mean_position']:.3f}
      Left periphery: {stats_6way['left_periphery']:.1%}
      Right periphery: {stats_6way['right_periphery']:.1%}
      L/R ratio: {stats_6way['lr_ratio']:.2f}
    """
    
    ax.text(0.1, 0.5, summary, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def main():
    print("="*70)
    print("DISCOURSE MARKER STATISTICS ANALYSIS")
    print("="*70)
    print(f"\n🏠 HOME: {HOME}")
    print(f"📁 OUTPUT_DIR: {OUTPUT_DIR}")
    
    # Load data
    print("\n📂 Loading data...")
    df_4way = load_data('4way')
    df_6way = load_data('6way')
    
    if df_4way is None or df_6way is None:
        print("\n❌ Error: Could not load data files")
        print(f"\n📝 Expected data location:")
        print(f"   {HOME}/data/iemocap_4way_data/train_4way_with_unified.csv")
        print(f"   {HOME}/data/iemocap_4way_data/val_4way_with_unified.csv")
        print(f"   {HOME}/data/iemocap_4way_data/test_4way_with_unified.csv")
        print(f"   (same for 6way)")
        return
    
    # Analyze markers
    print("\n🔍 Extracting and analyzing markers...")
    df_markers_4way, stats_4way = analyze_marker_statistics(df_4way, '4way')
    df_markers_6way, stats_6way = analyze_marker_statistics(df_6way, '6way')
    
    if df_markers_4way is None or df_markers_6way is None:
        print("❌ Error: No markers found")
        return
    
    # Print statistics
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'4-way':>20} {'6-way':>20}")
    print("-"*65)
    print(f"{'Total markers':<25} {stats_4way['total_markers']:>20} {stats_6way['total_markers']:>20}")
    print(f"{'Unique markers':<25} {stats_4way['unique_markers']:>20} {stats_6way['unique_markers']:>20}")
    print(f"{'Mean position':<25} {stats_4way['mean_position']:>20.3f} {stats_6way['mean_position']:>20.3f}")
    print(f"{'Left periphery':<25} {stats_4way['left_periphery']:>19.1%} {stats_6way['left_periphery']:>19.1%}")
    print(f"{'Right periphery':<25} {stats_4way['right_periphery']:>19.1%} {stats_6way['right_periphery']:>19.1%}")
    print(f"{'L/R ratio ⭐':<25} {stats_4way['lr_ratio']:>20.2f} {stats_6way['lr_ratio']:>20.2f}")
    
    print("\n" + "="*70)
    print("TOP 10 MARKERS (4-way)")
    print("="*70)
    for marker, count in list(stats_4way['marker_frequencies'].items())[:10]:
        pct = count / stats_4way['total_markers'] * 100
        pos_info = stats_4way['marker_positions'][marker]
        print(f"{marker:15s} {count:>5} ({pct:>5.1f}%)  pos={pos_info['mean_position']:.2f}  L/R={pos_info['lr_ratio']:.2f}")
    
    print("\n" + "="*70)
    print("EMOTION-SPECIFIC L/R RATIOS (4-way)")
    print("="*70)
    for emotion, stats in sorted(stats_4way['emotion_stats'].items()):
        print(f"{emotion:12s} L={stats['left_pct']:>5.1%}  R={stats['right_pct']:>5.1%}  L/R={stats['lr_ratio']:>5.2f}  (n={stats['count']})")
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    fig = plot_marker_analysis(df_markers_4way, df_markers_6way, stats_4way, stats_6way)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(OUTPUT_DIR / 'discourse_marker_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {OUTPUT_DIR}/discourse_marker_statistics.png")
    
    # Save detailed statistics
    report = {
        '4way': stats_4way,
        '6way': stats_6way
    }
    
    with open(OUTPUT_DIR / 'discourse_marker_statistics.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved report: {OUTPUT_DIR}/discourse_marker_statistics.json")
    
    # Save marker DataFrames
    df_markers_4way.to_csv(OUTPUT_DIR / 'markers_4way_extracted.csv', index=False)
    df_markers_6way.to_csv(OUTPUT_DIR / 'markers_6way_extracted.csv', index=False)
    print(f"✅ Saved marker data: {OUTPUT_DIR}/markers_*way_extracted.csv")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()