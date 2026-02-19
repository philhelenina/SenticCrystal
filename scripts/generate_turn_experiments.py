"""
Generate K-Turn Experiments from Pure Embeddings
===============================================

Create context windows for turn-level experiments from pure embeddings.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing.context_window import create_turn_dataset, save_turn_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
EMBEDDINGS_DIR = HOME_DIR / 'scripts' / 'embeddings'
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
RESULTS_DIR = HOME_DIR / 'results' / 'turn_experiments'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pure_embeddings(config_id, dataset_type):
    """Load pure embeddings and metadata."""
    
    # Load pure embeddings
    embeddings_file = EMBEDDINGS_DIR / f'X_textsroberta{dataset_type}_config{config_id}_pure.npy'
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Pure embeddings not found: {embeddings_file}")
    
    embeddings = np.load(embeddings_file)
    logger.info(f"Loaded pure embeddings: {embeddings.shape}")
    
    # Load corresponding CSV for metadata
    csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded metadata: {df.shape}")
    
    # Ensure embeddings and metadata align
    assert len(embeddings) == len(df), f"Mismatch: {len(embeddings)} embeddings vs {len(df)} metadata rows"
    
    return embeddings, df


def generate_k_turn_dataset(config_id, dataset_type, context_size):
    """Generate K-turn dataset for specific context size."""
    
    logger.info(f"Generating K={context_size} turn dataset for Config {config_id}, {dataset_type}")
    
    # Load pure embeddings and metadata
    embeddings, metadata = load_pure_embeddings(config_id, dataset_type)
    
    # Convert embeddings to list for context_window module
    embeddings_list = [emb for emb in embeddings]
    
    # Create turn dataset with context windows
    turn_dataset = create_turn_dataset(
        embeddings=embeddings_list,
        metadata=metadata,
        context_size=context_size,
        padding_mode='zero'
    )
    
    # Save the turn dataset
    output_file = RESULTS_DIR / f'X_textsroberta{dataset_type}_config{config_id}_k{context_size}.npy'
    
    # Load config details
    config_file = EMBEDDINGS_DIR / f"config{config_id}details{dataset_type}.json"
    config_details = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_details = json.load(f)
    
    save_turn_dataset(
        dataset=turn_dataset,
        output_path=output_file,
        config_details=config_details,
        context_size=context_size
    )
    
    logger.info(f"Generated K={context_size} dataset: {output_file}")
    return output_file


def generate_all_k_experiments(config_id, k_values=[0, 2, 4, 6], dataset_types=['train', 'val', 'test']):
    """Generate all K-turn experiments for a configuration."""
    
    logger.info(f"Generating all K-turn experiments for Config {config_id}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Dataset types: {dataset_types}")
    
    results = {}
    
    for dataset_type in dataset_types:
        results[dataset_type] = {}
        
        for k in k_values:
            try:
                if k == 0:
                    # K=0 means no context, just copy pure embeddings
                    output_file = generate_k0_dataset(config_id, dataset_type)
                else:
                    output_file = generate_k_turn_dataset(config_id, dataset_type, k)
                
                results[dataset_type][f'k{k}'] = str(output_file)
                
            except Exception as e:
                logger.error(f"Failed to generate K={k} for {dataset_type}: {str(e)}")
                results[dataset_type][f'k{k}'] = None
    
    # Save summary
    summary_file = RESULTS_DIR / f'config{config_id}_turn_experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Turn experiments summary saved: {summary_file}")
    return results


def generate_k0_dataset(config_id, dataset_type):
    """Generate K=0 dataset (no context)."""
    
    logger.info(f"Generating K=0 dataset for Config {config_id}, {dataset_type}")
    
    # Load pure embeddings and metadata
    embeddings, metadata = load_pure_embeddings(config_id, dataset_type)
    
    # For K=0, each "context window" is just the single embedding
    single_embeddings = [[emb] for emb in embeddings]
    
    # Create dataset structure
    features_dict = {
        'text_features': single_embeddings,
        'ids': metadata['id'].tolist(),
        'file_id': metadata['file_id'].tolist(),
        'utterance_num': metadata['utterance_num'].tolist(),
        'label': metadata['label'].tolist()
    }
    
    k0_dataset = pd.DataFrame.from_dict(features_dict)
    
    # Save K=0 dataset
    output_file = RESULTS_DIR / f'X_textsroberta{dataset_type}_config{config_id}_k0.npy'
    
    # Load config details
    config_file = EMBEDDINGS_DIR / f"config{config_id}details{dataset_type}.json"
    config_details = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_details = json.load(f)
    
    save_turn_dataset(
        dataset=k0_dataset,
        output_path=output_file,
        config_details=config_details,
        context_size=0
    )
    
    logger.info(f"Generated K=0 dataset: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate K-turn experiments from pure embeddings")
    parser.add_argument('--config_id', type=int, required=True, help='Configuration ID')
    parser.add_argument('--k_values', nargs='+', type=int, default=[0, 2, 4, 6], 
                       help='Context sizes to generate (default: 0 2 4 6)')
    parser.add_argument('--dataset_types', nargs='+', default=['train', 'val', 'test'],
                       help='Dataset types to process (default: train val test)')
    parser.add_argument('--single_k', type=int, help='Generate only single K value')
    parser.add_argument('--single_dataset', help='Generate only single dataset type')
    
    args = parser.parse_args()
    
    # Validate that pure embeddings exist
    for dataset_type in args.dataset_types:
        embeddings_file = EMBEDDINGS_DIR / f'X_textsroberta{dataset_type}_config{args.config_id}_pure.npy'
        if not embeddings_file.exists():
            logger.error(f"Pure embeddings not found: {embeddings_file}")
            logger.error("Please run embeddings_pure.py first to generate pure embeddings")
            return
    
    if args.single_k is not None and args.single_dataset:
        # Generate single K for single dataset
        generate_k_turn_dataset(args.config_id, args.single_dataset, args.single_k)
    
    elif args.single_k is not None:
        # Generate single K for all datasets
        for dataset_type in args.dataset_types:
            if args.single_k == 0:
                generate_k0_dataset(args.config_id, dataset_type)
            else:
                generate_k_turn_dataset(args.config_id, dataset_type, args.single_k)
    
    elif args.single_dataset:
        # Generate all K for single dataset
        for k in args.k_values:
            if k == 0:
                generate_k0_dataset(args.config_id, args.single_dataset)
            else:
                generate_k_turn_dataset(args.config_id, args.single_dataset, k)
    
    else:
        # Generate all combinations
        generate_all_k_experiments(args.config_id, args.k_values, args.dataset_types)
    
    logger.info("Turn experiment generation completed.")


if __name__ == "__main__":
    main()

# Examples:
# python generate_turn_experiments.py --config_id 146
# python generate_turn_experiments.py --config_id 146 --k_values 0 2 6
# python generate_turn_experiments.py --config_id 146 --single_k 2 --single_dataset train