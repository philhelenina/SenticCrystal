"""
Real Cumulative Context Creation for IEMOCAP
===========================================

Create actual cumulative context windows based on dialogue structure,
not just weighted combinations of different K-values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
EMBEDDINGS_DIR = HOME_DIR / 'scripts' / 'embeddings'
RESULTS_DIR = HOME_DIR / 'results' / 'cumulative_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dialogue_structure(dataset_type='train'):
    """Analyze actual dialogue structure in IEMOCAP data."""
    
    logger.info(f"Analyzing dialogue structure for {dataset_type} set")
    
    csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    df = pd.read_csv(csv_file)
    
    # Extract dialogue IDs from filename (e.g., 'Ses01F_impro01_F000' -> 'Ses01F_impro01')
    df['dialogue_id'] = df['filename'].str.extract(r'(Ses\d+[MF]_[^_]+)')
    
    # Analyze dialogue lengths
    dialogue_lengths = df.groupby('dialogue_id').size().to_dict()
    
    logger.info(f"Found {len(dialogue_lengths)} dialogues")
    logger.info(f"Average dialogue length: {np.mean(list(dialogue_lengths.values())):.1f}")
    logger.info(f"Max dialogue length: {max(dialogue_lengths.values())}")
    logger.info(f"Min dialogue length: {min(dialogue_lengths.values())}")
    
    # Distribution of dialogue lengths
    length_dist = defaultdict(int)
    for length in dialogue_lengths.values():
        length_dist[length] += 1
    
    logger.info("Dialogue length distribution:")
    for length in sorted(length_dist.keys())[:10]:  # Show first 10
        logger.info(f"  Length {length}: {length_dist[length]} dialogues")
    
    return df, dialogue_lengths


def create_cumulative_context_windows(df, embeddings, max_context_size=10):
    """Create real cumulative context windows based on dialogue structure."""
    
    logger.info(f"Creating cumulative context windows (max_context_size={max_context_size})")
    
    cumulative_data = []
    
    # Group by dialogue
    for dialogue_id, dialogue_group in df.groupby('dialogue_id'):
        dialogue_group = dialogue_group.sort_index()  # Ensure chronological order
        
        logger.debug(f"Processing dialogue {dialogue_id} with {len(dialogue_group)} utterances")
        
        for i, (idx, row) in enumerate(dialogue_group.iterrows()):
            # Create cumulative context for this utterance
            # Include all previous utterances in the dialogue up to max_context_size
            
            start_idx = max(0, i - max_context_size + 1)
            context_indices = dialogue_group.iloc[start_idx:i+1].index.tolist()
            
            # Get embeddings for context
            context_embeddings = embeddings[context_indices]
            
            # Pad if necessary (for consistent tensor shapes)
            if len(context_embeddings) < max_context_size:
                # Pad with zeros at the beginning
                padding_size = max_context_size - len(context_embeddings)
                padding = np.zeros((padding_size, context_embeddings.shape[1]))
                context_embeddings = np.vstack([padding, context_embeddings])
            
            cumulative_data.append({
                'original_index': idx,
                'dialogue_id': dialogue_id,
                'position_in_dialogue': i,
                'context_size': min(i + 1, max_context_size),
                'actual_context_size': len(context_indices),
                'context_embeddings': context_embeddings,
                'label': row['label'],
                'filename': row['filename']
            })
    
    logger.info(f"Created {len(cumulative_data)} cumulative context windows")
    
    return cumulative_data


def save_cumulative_context_data(cumulative_data, config_id, dataset_type, max_context_size):
    """Save cumulative context data for training."""
    
    # Separate embeddings and metadata
    all_embeddings = np.stack([item['context_embeddings'] for item in cumulative_data])
    
    # Save embeddings
    embeddings_file = EMBEDDINGS_DIR / f'X_cumulative_k{max_context_size}_{dataset_type}_config{config_id}.npy'
    np.save(embeddings_file, all_embeddings)
    
    # Save metadata
    metadata = []
    for item in cumulative_data:
        metadata.append({
            'original_index': item['original_index'],
            'dialogue_id': item['dialogue_id'],
            'position_in_dialogue': item['position_in_dialogue'],
            'context_size': item['context_size'],
            'actual_context_size': item['actual_context_size'],
            'label': item['label'],
            'filename': item['filename']
        })
    
    metadata_file = RESULTS_DIR / f'cumulative_metadata_k{max_context_size}_{dataset_type}_config{config_id}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved cumulative embeddings: {embeddings_file}")
    logger.info(f"Saved cumulative metadata: {metadata_file}")
    logger.info(f"Shape: {all_embeddings.shape}")
    
    return embeddings_file, metadata_file


def analyze_cumulative_context_statistics(cumulative_data):
    """Analyze statistics of cumulative context windows."""
    
    logger.info("Analyzing cumulative context statistics")
    
    context_sizes = [item['context_size'] for item in cumulative_data]
    actual_sizes = [item['actual_context_size'] for item in cumulative_data]
    positions = [item['position_in_dialogue'] for item in cumulative_data]
    
    stats = {
        'total_samples': len(cumulative_data),
        'context_size_distribution': {
            'mean': np.mean(context_sizes),
            'std': np.std(context_sizes),
            'min': np.min(context_sizes),
            'max': np.max(context_sizes)
        },
        'position_distribution': {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'min': np.min(positions),
            'max': np.max(positions)
        },
        'context_utilization': {
            f'size_{i}': context_sizes.count(i) for i in range(1, 11)
        }
    }
    
    logger.info(f"Context size stats: mean={stats['context_size_distribution']['mean']:.2f}, "
               f"std={stats['context_size_distribution']['std']:.2f}")
    
    logger.info("Context size utilization:")
    for size, count in stats['context_utilization'].items():
        percentage = (count / len(cumulative_data)) * 100
        logger.info(f"  {size}: {count} samples ({percentage:.1f}%)")
    
    return stats


def create_progressive_cumulative_strategies(config_id):
    """Create multiple cumulative strategies with different max context sizes."""
    
    logger.info("="*80)
    logger.info("CREATING PROGRESSIVE CUMULATIVE STRATEGIES")
    logger.info("="*80)
    
    strategies = {
        'short_cumulative': {'max_context': 3, 'description': 'Short-term cumulative context'},
        'medium_cumulative': {'max_context': 6, 'description': 'Medium-term cumulative context'}, 
        'long_cumulative': {'max_context': 10, 'description': 'Long-term cumulative context'},
        'full_cumulative': {'max_context': 20, 'description': 'Full dialogue cumulative context'}
    }
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"STRATEGY: {strategy_name}")
        logger.info(f"Description: {strategy_config['description']}")
        logger.info(f"Max context size: {strategy_config['max_context']}")
        logger.info(f"{'='*50}")
        
        strategy_results = {}
        
        for dataset_type in ['train', 'val', 'test']:
            logger.info(f"\nProcessing {dataset_type} set for {strategy_name}")
            
            # Analyze dialogue structure
            df, dialogue_lengths = analyze_dialogue_structure(dataset_type)
            
            # Load embeddings
            embedding_file = EMBEDDINGS_DIR / f'X_textsroberta{dataset_type}_config{config_id}_pure.npy'
            embeddings = np.load(embedding_file)
            
            # Create cumulative context
            cumulative_data = create_cumulative_context_windows(
                df, embeddings, max_context_size=strategy_config['max_context']
            )
            
            # Analyze statistics
            stats = analyze_cumulative_context_statistics(cumulative_data)
            
            # Save data
            embeddings_file, metadata_file = save_cumulative_context_data(
                cumulative_data, config_id, dataset_type, strategy_config['max_context']
            )
            
            strategy_results[dataset_type] = {
                'embeddings_file': str(embeddings_file),
                'metadata_file': str(metadata_file),
                'statistics': stats,
                'dialogue_lengths': dialogue_lengths
            }
        
        results[strategy_name] = {
            'config': strategy_config,
            'datasets': strategy_results
        }
    
    # Save comprehensive results
    results_file = RESULTS_DIR / f'cumulative_strategies_config{config_id}.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info(f"\nComprehensive results saved: {results_file}")
    
    return results


def main():
    config_id = 146
    
    logger.info("="*80)
    logger.info("REAL CUMULATIVE CONTEXT CREATION")
    logger.info("="*80)
    logger.info(f"Config ID: {config_id}")
    
    # Create progressive cumulative strategies
    results = create_progressive_cumulative_strategies(config_id)
    
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for strategy_name, strategy_data in results.items():
        max_context = strategy_data['config']['max_context']
        logger.info(f"\n{strategy_name}: max_context={max_context}")
        
        for dataset_type in ['train', 'val', 'test']:
            stats = strategy_data['datasets'][dataset_type]['statistics']
            logger.info(f"  {dataset_type}: {stats['total_samples']} samples, "
                       f"avg_context={stats['context_size_distribution']['mean']:.1f}")
    
    logger.info("\n✅ Real cumulative context creation completed!")
    logger.info("Now you can train classifiers on these cumulative context windows.")


if __name__ == "__main__":
    main()