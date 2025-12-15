#!/usr/bin/env python3
"""
generate_tmlr_figures.py
- TMLR 논문용 개별 figure 생성
- 각 그림을 PDF와 PNG 형식으로 저장
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter

# Set publication quality
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_style("whitegrid")

# Paths
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
DATA_DIR_4WAY = HOME / "data" / "iemocap_4way_data"
DATA_DIR_6WAY = HOME / "data" / "iemocap_6way_data"
OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_figure(filename_base):
    """Save figure in both PDF and PNG formats"""
    pdf_path = OUTPUT_DIR / f'{filename_base}.pdf'
    png_path = OUTPUT_DIR / f'{filename_base}.png'
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved PDF: {pdf_path}")
    print(f"✅ Saved PNG: {png_path}")

def simple_sent_split(text: str):
    """Hierarchical embedding과 동일한 문장 분리"""
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def load_data(task='4way'):
    """Load IEMOCAP data"""
    # Select appropriate directory based on task
    DATA_DIR = DATA_DIR_4WAY if task == '4way' else DATA_DIR_6WAY
    
    all_dfs = []
    for split in ['train', 'val', 'test']:
        file_path = DATA_DIR / f'{split}_{task}_unified.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['split'] = split
            all_dfs.append(df)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if all_dfs:
        print(f"✅ Loaded {task} data: {len(all_dfs)} splits, {sum(len(d) for d in all_dfs)} total samples")
        return pd.concat(all_dfs, ignore_index=True)
    return None

def fig1_dialogue_length_histogram(df):
    """Figure 1a: Dialogue length distribution"""
    plt.figure(figsize=(6, 4))
    
    # Group by dialogue
    dialogue_lengths = df.groupby('file_id')['utterance_num'].max() + 1
    
    # Histogram
    plt.hist(dialogue_lengths, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add p95 line
    p95 = dialogue_lengths.quantile(0.95)
    plt.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'p95={p95:.0f}')
    
    # Add K=100 line
    plt.axvline(100, color='orange', linestyle='--', linewidth=2, label='K=100')
    
    plt.xlabel('Dialogue Length (utterances)')
    plt.ylabel('Frequency')
    plt.title('Dialogue Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save both formats
    save_figure('fig1a_dialogue_length')
    plt.close()
    
    # Return stats for text
    return {
        'mean': dialogue_lengths.mean(),
        'p95': p95,
        'k100_coverage': (dialogue_lengths <= 100).mean() * 100
    }

def fig2_sentence_distribution(df):
    """Figure 1b: Sentence distribution"""
    plt.figure(figsize=(6, 4))
    
    # Get text column
    text_col = None
    for col in ['utterance', 'text', 'transcript']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("⚠️ No text column found")
        return None
    
    # Count sentences
    sentence_counts = []
    for text in df[text_col].astype(str).fillna(""):
        if not text or text == 'nan':
            sents = []
        else:
            sents = simple_sent_split(text)
            if not sents:
                sents = [text.strip()]
        sentence_counts.append(len(sents))
    
    sentence_counts = np.array(sentence_counts)
    
    # Histogram
    max_val = min(12, sentence_counts.max())
    bins = range(0, max_val+2)
    plt.hist(sentence_counts, bins=bins, alpha=0.7, color='coral', edgecolor='black')
    
    # Highlight single-sentence
    single_pct = (sentence_counts == 1).mean() * 100
    plt.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    plt.text(1.5, plt.ylim()[1]*0.9, f'{single_pct:.1f}% single-sentence', 
             fontsize=11, color='red')
    
    plt.xlabel('Number of Sentences per Utterance')
    plt.ylabel('Frequency')
    plt.title('Sentence Distribution (Hierarchical Segmentation)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save both formats
    save_figure('fig1b_sentence_distribution')
    plt.close()
    
    return {
        'single_pct': single_pct,
        'multi_pct': (sentence_counts > 1).mean() * 100,
        'mean': sentence_counts.mean(),
        'max': sentence_counts.max()
    }

def fig3_emotion_violin_4way(df):
    """Figure 1c: 4-way emotion-specific patterns"""
    plt.figure(figsize=(6, 4))
    
    # Filter valid labels
    valid_df = df[df['label_num'] >= 0].copy()
    
    # Emotion mapping for 4-way
    emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral'}
    
    # Prepare data for violin plot
    emotion_data = []
    emotion_labels = []
    
    for label_num in sorted(valid_df['label_num'].unique())[:4]:  # Only first 4 for 4-way
        emo_df = valid_df[valid_df['label_num'] == label_num]
        dialogue_lengths = emo_df.groupby('file_id')['utterance_num'].max() + 1
        
        if len(dialogue_lengths) > 0:
            emotion_data.append(dialogue_lengths.values)
            emotion_labels.append(emotion_map.get(int(label_num), str(label_num)))
    
    # Violin plot
    parts = plt.violinplot(emotion_data, positions=range(len(emotion_data)), 
                           showmeans=True, showmedians=True)
    
    # Customize colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    plt.xticks(range(len(emotion_labels)), emotion_labels)
    plt.ylabel('Dialogue Length (utterances)')
    plt.title('4-way: Emotion-Specific Dialogue Patterns')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save both formats
    save_figure('fig1c_emotion_4way')
    plt.close()

def fig4_emotion_violin_6way(df):
    """Figure 1d: 6-way emotion-specific patterns"""
    plt.figure(figsize=(6, 4))  # Same size as 4-way for consistency
    
    # Filter valid labels
    valid_df = df[df['label_num'] >= 0].copy()
    
    # Emotion mapping for 6-way
    emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral', 4: 'Excited', 5: 'Frustrated'}
    
    # Prepare data for violin plot
    emotion_data = []
    emotion_labels = []
    
    for label_num in sorted(valid_df['label_num'].unique())[:6]:  # All 6 for 6-way
        emo_df = valid_df[valid_df['label_num'] == label_num]
        dialogue_lengths = emo_df.groupby('file_id')['utterance_num'].max() + 1
        
        if len(dialogue_lengths) > 0:
            emotion_data.append(dialogue_lengths.values)
            emotion_labels.append(emotion_map.get(int(label_num), str(label_num)))
    
    # Violin plot
    parts = plt.violinplot(emotion_data, positions=range(len(emotion_data)), 
                           showmeans=True, showmedians=True)
    
    # Customize colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6', '#FFA07A', '#DDA0DD']
    for pc, color in zip(parts['bodies'], colors[:len(parts['bodies'])]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    plt.xticks(range(len(emotion_labels)), emotion_labels)
    plt.ylabel('Dialogue Length (utterances)')
    plt.title('6-way: Emotion-Specific Dialogue Patterns')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save both formats
    save_figure('fig1d_emotion_6way')
    plt.close()

def fig5_discourse_position(df):
    """Figure 2a: Discourse marker position distribution"""
    plt.figure(figsize=(6, 4))
    
    # Simple discourse markers
    markers = ['well', 'so', 'but', 'oh', 'and', 'like', 'actually', 'really']
    
    positions = []
    text_col = None
    for col in ['utterance', 'text', 'transcript']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col:
        for text in df[text_col].dropna():
            text_lower = str(text).lower()
            tokens = text_lower.split()
            
            for i, token in enumerate(tokens):
                if any(token.startswith(m) for m in markers):
                    rel_pos = i / max(len(tokens) - 1, 1)
                    positions.append(rel_pos)
    
    if positions:
        # Histogram
        plt.hist(positions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mark peripheries
        plt.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Left periphery')
        plt.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='Right periphery')
        
        # Calculate L/R ratio
        positions = np.array(positions)
        left = (positions < 0.1).mean()
        right = (positions > 0.9).mean()
        lr_ratio = left / max(right, 0.001)
        
        plt.xlabel('Relative Position in Utterance')
        plt.ylabel('Frequency')
        plt.title(f'Discourse Marker Distribution (L/R = {lr_ratio:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save both formats
        save_figure('fig2a_discourse_position')
        plt.close()
        
        return {'lr_ratio': lr_ratio, 'left_pct': left*100, 'right_pct': right*100}
    
    return None

def fig6_lr_ratio_by_emotion(df):
    """Figure 2b: L/R ratio by emotion"""
    plt.figure(figsize=(6, 5))
    
    # Simple discourse markers
    markers = ['well', 'so', 'but', 'oh', 'and', 'like', 'actually', 'really']
    
    text_col = None
    for col in ['utterance', 'text', 'transcript']:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        return None
    
    # Collect positions by emotion
    emotion_positions = {0: [], 1: [], 2: [], 3: []}  # 4-way emotions
    emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral'}
    
    valid_df = df[df['label_num'].isin([0, 1, 2, 3])]
    
    for _, row in valid_df.iterrows():
        text = str(row[text_col]).lower()
        tokens = text.split()
        emotion = int(row['label_num'])
        
        for i, token in enumerate(tokens):
            if any(token.startswith(m) for m in markers):
                rel_pos = i / max(len(tokens) - 1, 1)
                emotion_positions[emotion].append(rel_pos)
    
    # Calculate L/R ratios
    lr_ratios = []
    emotions = []
    
    for emo_id, positions in emotion_positions.items():
        if len(positions) > 0:
            positions = np.array(positions)
            left = (positions < 0.1).mean()
            right = (positions > 0.9).mean()
            lr = left / max(right, 0.001)
            lr_ratios.append(lr)
            emotions.append(emotion_map[emo_id])
    
    # Horizontal bar chart
    y_pos = np.arange(len(emotions))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
    
    bars = plt.barh(y_pos, lr_ratios, color=colors[:len(emotions)], alpha=0.7)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, lr_ratios)):
        plt.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center')
    
    # Overall mean line
    mean_lr = np.mean(lr_ratios)
    plt.axvline(mean_lr, color='red', linestyle='--', alpha=0.5, label=f'Mean = {mean_lr:.2f}')
    
    plt.yticks(y_pos, emotions)
    plt.xlabel('Left/Right Periphery Ratio')
    plt.title('Emotion-Specific Discourse Marker Positioning')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save both formats
    save_figure('fig2b_lr_ratio_emotion')
    plt.close()

def main():
    print("="*70)
    print("GENERATING TMLR FIGURES (PDF + PNG)")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    df_4way = load_data('4way')
    df_6way = load_data('6way')
    
    if df_4way is None:
        print("❌ Could not load 4-way data")
        return
    
    # Generate figures
    print("\n📊 Generating figures...")
    
    # Figure 1a: Dialogue length
    print("\n[1/6] Dialogue length histogram...")
    stats1 = fig1_dialogue_length_histogram(df_4way)
    if stats1:
        print(f"  Mean: {stats1['mean']:.1f}, p95: {stats1['p95']:.0f}")
        print(f"  K=100 coverage: {stats1['k100_coverage']:.1f}%")
    
    # Figure 1b: Sentence distribution
    print("\n[2/6] Sentence distribution...")
    stats2 = fig2_sentence_distribution(df_4way)
    if stats2:
        print(f"  Single-sentence: {stats2['single_pct']:.1f}%")
        print(f"  Multi-sentence: {stats2['multi_pct']:.1f}%")
    
    # Figure 1c: 4-way emotions
    print("\n[3/6] 4-way emotion patterns...")
    fig3_emotion_violin_4way(df_4way)
    
    # Figure 1d: 6-way emotions
    if df_6way is not None:
        print("\n[4/6] 6-way emotion patterns...")
        fig4_emotion_violin_6way(df_6way)
    else:
        print("\n[4/6] ⚠️ Skipping 6-way (data not found)")
    
    # Figure 2a: Discourse positions
    print("\n[5/6] Discourse marker positions...")
    stats5 = fig5_discourse_position(df_4way)
    if stats5:
        print(f"  L/R ratio: {stats5['lr_ratio']:.2f}")
        print(f"  Left periphery: {stats5['left_pct']:.1f}%")
    
    # Figure 2b: L/R by emotion
    print("\n[6/6] L/R ratio by emotion...")
    fig6_lr_ratio_by_emotion(df_4way)
    
    print(f"\n✅ All figures saved to: {OUTPUT_DIR}")
    print("   - PDF format (for LaTeX/publication)")
    print("   - PNG format (for presentations/web)")
    
    print("\n📝 For LaTeX (use PDF):")
    print("""
\\begin{figure}[t]
    \\centering
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{figures/fig1a_dialogue_length.pdf}
        \\caption{Dialogue length distribution}
    \\end{subfigure}
    \\hfill
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{figures/fig1b_sentence_distribution.pdf}
        \\caption{Sentence distribution}
    \\end{subfigure}
    \\vskip\\baselineskip
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{figures/fig1c_emotion_4way.pdf}
        \\caption{4-way emotion patterns}
    \\end{subfigure}
    \\hfill
    \\begin{subfigure}[b]{0.48\\textwidth}
        \\includegraphics[width=\\textwidth]{figures/fig1d_emotion_6way.pdf}
        \\caption{6-way emotion patterns}
    \\end{subfigure}
    \\caption{IEMOCAP dataset characteristics.}
    \\label{fig:dataset}
\\end{figure}
    """)

if __name__ == "__main__":
    main()