"""
IEMOCAP Dialogue Structure Analysis
==================================

Analyze actual dialogue structure, conversation lengths, and speaker patterns
in IEMOCAP data to understand realistic cumulative context requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
RESULTS_DIR = HOME_DIR / 'results' / 'dialogue_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_dialogue_info(df):
    """Extract dialogue and speaker information from filenames."""
    
    # Use file_id as dialogue_id (e.g., 'Ses01F_impro01')
    df['dialogue_id'] = df['file_id']
    
    # Extract speaker ID from id column (e.g., 'Ses01F_impro01_F000' -> 'F')
    df['speaker'] = df['id'].str.extract(r'Ses\d+[MF]_[^_]+_([MF])\d+')
    
    # Use existing utterance_num as position in dialogue
    df['utt_num_in_dialogue'] = df['utterance_num']
    
    logger.info(f"Extracted dialogue info:")
    logger.info(f"  Unique dialogues: {df['dialogue_id'].nunique()}")
    logger.info(f"  Unique speakers: {df['speaker'].unique()}")
    logger.info(f"  Sample dialogue ID: {df['dialogue_id'].iloc[0]}")
    
    return df


def analyze_dialogue_lengths(df):
    """Analyze dialogue lengths and distribution."""
    
    logger.info("Analyzing dialogue lengths...")
    
    # Group by dialogue and count utterances
    dialogue_lengths = df.groupby('dialogue_id').size()
    
    # Get statistics
    stats = {
        'total_dialogues': len(dialogue_lengths),
        'total_utterances': len(df),
        'mean_length': dialogue_lengths.mean(),
        'median_length': dialogue_lengths.median(),
        'std_length': dialogue_lengths.std(),
        'min_length': dialogue_lengths.min(),
        'max_length': dialogue_lengths.max(),
        'percentiles': {
            '25th': dialogue_lengths.quantile(0.25),
            '50th': dialogue_lengths.quantile(0.50),
            '75th': dialogue_lengths.quantile(0.75),
            '90th': dialogue_lengths.quantile(0.90),
            '95th': dialogue_lengths.quantile(0.95)
        }
    }
    
    logger.info(f"Total dialogues: {stats['total_dialogues']}")
    logger.info(f"Total utterances: {stats['total_utterances']}")
    logger.info(f"Average dialogue length: {stats['mean_length']:.1f} utterances")
    logger.info(f"Median dialogue length: {stats['median_length']:.1f} utterances")
    logger.info(f"Min-Max length: {stats['min_length']}-{stats['max_length']} utterances")
    
    logger.info("Length percentiles:")
    for pct, value in stats['percentiles'].items():
        logger.info(f"  {pct}: {value:.1f} utterances")
    
    return dialogue_lengths, stats


def analyze_speaker_patterns(df):
    """Analyze speaker turn-taking patterns."""
    
    logger.info("Analyzing speaker patterns...")
    
    speaker_stats = {}
    
    for dialogue_id, dialogue_group in df.groupby('dialogue_id'):
        dialogue_group = dialogue_group.sort_values('utt_num_in_dialogue')
        
        speakers = dialogue_group['speaker'].tolist()
        
        # Count speaker turns
        speaker_counts = Counter(speakers)
        
        # Analyze turn-taking patterns
        turns = []
        current_speaker = speakers[0]
        turn_length = 1
        
        for speaker in speakers[1:]:
            if speaker == current_speaker:
                turn_length += 1
            else:
                turns.append((current_speaker, turn_length))
                current_speaker = speaker
                turn_length = 1
        turns.append((current_speaker, turn_length))
        
        speaker_stats[dialogue_id] = {
            'length': len(speakers),
            'speakers': list(speaker_counts.keys()),
            'speaker_counts': dict(speaker_counts),
            'turns': turns,
            'avg_turn_length': np.mean([t[1] for t in turns]),
            'num_turns': len(turns)
        }
    
    # Aggregate statistics
    all_turn_lengths = []
    all_num_turns = []
    speaker_balance = []
    
    for dialogue_id, stats in speaker_stats.items():
        all_turn_lengths.extend([t[1] for t in stats['turns']])
        all_num_turns.append(stats['num_turns'])
        
        # Calculate speaker balance (ratio of utterances)
        counts = list(stats['speaker_counts'].values())
        if len(counts) == 2:
            balance = min(counts) / max(counts)
            speaker_balance.append(balance)
    
    summary = {
        'avg_turn_length': np.mean(all_turn_lengths),
        'median_turn_length': np.median(all_turn_lengths),
        'avg_num_turns_per_dialogue': np.mean(all_num_turns),
        'avg_speaker_balance': np.mean(speaker_balance) if speaker_balance else 0
    }
    
    logger.info(f"Average turn length: {summary['avg_turn_length']:.2f} utterances")
    logger.info(f"Median turn length: {summary['median_turn_length']:.1f} utterances")
    logger.info(f"Average turns per dialogue: {summary['avg_num_turns_per_dialogue']:.1f}")
    logger.info(f"Average speaker balance: {summary['avg_speaker_balance']:.3f}")
    
    return speaker_stats, summary


def analyze_label_distribution(df):
    """Analyze emotion label distribution."""
    
    logger.info("Analyzing emotion label distribution...")
    
    # Overall distribution
    overall_dist = df['label'].value_counts()
    logger.info("Overall label distribution:")
    for label, count in overall_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Distribution by dialogue
    dialogue_label_stats = {}
    for dialogue_id, dialogue_group in df.groupby('dialogue_id'):
        # Only consider valid labels (not -1)
        valid_labels = dialogue_group[dialogue_group['label'] != '-1']
        
        if len(valid_labels) > 0:
            label_dist = valid_labels['label'].value_counts()
            dialogue_label_stats[dialogue_id] = {
                'total_utterances': len(dialogue_group),
                'labeled_utterances': len(valid_labels),
                'label_distribution': dict(label_dist),
                'dominant_emotion': label_dist.index[0] if len(label_dist) > 0 else None
            }
    
    # Calculate label diversity per dialogue
    label_diversities = []
    for stats in dialogue_label_stats.values():
        num_unique_labels = len(stats['label_distribution'])
        label_diversities.append(num_unique_labels)
    
    logger.info(f"Average unique emotions per dialogue: {np.mean(label_diversities):.2f}")
    logger.info(f"Max unique emotions in single dialogue: {max(label_diversities)}")
    
    return dialogue_label_stats


def analyze_cumulative_context_potential(df, dialogue_lengths):
    """Analyze potential for cumulative context at different positions."""
    
    logger.info("Analyzing cumulative context potential...")
    
    context_analysis = {}
    
    for k in [1, 2, 3, 5, 8, 10, 15, 20]:
        context_analysis[f'k{k}'] = {
            'positions_with_full_context': 0,
            'positions_with_partial_context': 0,
            'total_positions': 0,
            'avg_context_size': []
        }
    
    for dialogue_id, dialogue_group in df.groupby('dialogue_id'):
        dialogue_length = len(dialogue_group)
        
        for position in range(dialogue_length):
            for k in [1, 2, 3, 5, 8, 10, 15, 20]:
                key = f'k{k}'
                
                # Calculate available context at this position
                available_context = min(position + 1, k)
                
                context_analysis[key]['total_positions'] += 1
                context_analysis[key]['avg_context_size'].append(available_context)
                
                if available_context == k:
                    context_analysis[key]['positions_with_full_context'] += 1
                else:
                    context_analysis[key]['positions_with_partial_context'] += 1
    
    # Calculate utilization percentages
    logger.info("Cumulative context utilization:")
    for k in [1, 2, 3, 5, 8, 10, 15, 20]:
        key = f'k{k}'
        stats = context_analysis[key]
        total = stats['total_positions']
        full_pct = (stats['positions_with_full_context'] / total) * 100
        avg_size = np.mean(stats['avg_context_size'])
        
        logger.info(f"  K={k:2d}: {full_pct:5.1f}% full context, avg size: {avg_size:.1f}")
    
    return context_analysis


def create_visualizations(dialogue_lengths, df, save_dir):
    """Create visualizations of dialogue structure."""
    
    logger.info("Creating visualizations...")
    
    # 1. Dialogue length distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(dialogue_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Dialogue Length (utterances)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dialogue Lengths')
    plt.axvline(dialogue_lengths.mean(), color='red', linestyle='--', label=f'Mean: {dialogue_lengths.mean():.1f}')
    plt.axvline(dialogue_lengths.median(), color='orange', linestyle='--', label=f'Median: {dialogue_lengths.median():.1f}')
    plt.legend()
    
    # 2. Cumulative dialogue length
    plt.subplot(2, 2, 2)
    sorted_lengths = sorted(dialogue_lengths)
    cumulative_pct = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cumulative_pct)
    plt.xlabel('Dialogue Length (utterances)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Distribution of Dialogue Lengths')
    plt.grid(True, alpha=0.3)
    
    # 3. Label distribution
    plt.subplot(2, 2, 3)
    label_counts = df[df['label'] != '-1']['label'].value_counts()
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
    plt.title('Emotion Label Distribution')
    
    # 4. Context utilization
    plt.subplot(2, 2, 4)
    k_values = [1, 2, 3, 5, 8, 10, 15, 20]
    utilization = []
    
    for k in k_values:
        full_context_count = 0
        total_count = 0
        
        for dialogue_length in dialogue_lengths:
            for position in range(dialogue_length):
                total_count += 1
                if position + 1 >= k:
                    full_context_count += 1
        
        utilization.append((full_context_count / total_count) * 100)
    
    plt.plot(k_values, utilization, marker='o')
    plt.xlabel('Context Size (K)')
    plt.ylabel('Full Context Utilization (%)')
    plt.title('Context Size Utilization')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dialogue_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {save_dir / 'dialogue_analysis.png'}")


def main():
    logger.info("="*80)
    logger.info("IEMOCAP DIALOGUE STRUCTURE ANALYSIS")
    logger.info("="*80)
    
    results = {}
    
    for dataset_type in ['train', 'val', 'test']:
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING {dataset_type.upper()} SET")
        logger.info(f"{'='*60}")
        
        # Load data
        csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
        df = pd.read_csv(csv_file)
        
        logger.info(f"Loaded {len(df)} utterances from {csv_file}")
        
        # Extract dialogue information
        df = extract_dialogue_info(df)
        
        # Analyze dialogue lengths
        dialogue_lengths, length_stats = analyze_dialogue_lengths(df)
        
        # Analyze speaker patterns
        speaker_stats, speaker_summary = analyze_speaker_patterns(df)
        
        # Analyze label distribution
        label_stats = analyze_label_distribution(df)
        
        # Analyze cumulative context potential
        context_analysis = analyze_cumulative_context_potential(df, dialogue_lengths)
        
        # Create visualizations for train set
        if dataset_type == 'train':
            create_visualizations(dialogue_lengths, df, RESULTS_DIR)
        
        # Store results
        results[dataset_type] = {
            'length_stats': length_stats,
            'speaker_summary': speaker_summary,
            'context_analysis': context_analysis,
            'total_utterances': len(df),
            'total_dialogues': len(dialogue_lengths),
            'dialogue_lengths': dialogue_lengths.tolist()
        }
    
    # Save comprehensive analysis
    import json
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results_file = RESULTS_DIR / 'comprehensive_dialogue_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info(f"\nComprehensive analysis saved: {results_file}")
    
    # Print summary recommendations
    logger.info("\n" + "="*80)
    logger.info("CUMULATIVE CONTEXT RECOMMENDATIONS")
    logger.info("="*80)
    
    train_stats = results['train']['length_stats']
    
    logger.info(f"Based on dialogue length analysis:")
    logger.info(f"- Average dialogue length: {train_stats['mean_length']:.1f} utterances")
    logger.info(f"- 75th percentile length: {train_stats['percentiles']['75th']:.1f} utterances")
    logger.info(f"- 90th percentile length: {train_stats['percentiles']['90th']:.1f} utterances")
    
    logger.info(f"\nRecommended cumulative context sizes:")
    logger.info(f"- Short context (K=3): Covers early dialogue positions well")
    logger.info(f"- Medium context (K=6): Balances coverage and context richness")
    logger.info(f"- Long context (K=10): Captures most dialogue patterns")
    logger.info(f"- Full context (K=20): Ensures complete dialogue coverage")
    
    logger.info("\n✅ Dialogue structure analysis completed!")


if __name__ == "__main__":
    main()