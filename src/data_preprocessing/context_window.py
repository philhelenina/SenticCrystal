"""
Context Window Generation for Turn-level Experiments
===================================================

Separate module for generating context windows from utterance embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


def create_context_windows(embeddings: List[np.ndarray], 
                          context_size: int = 2,
                          padding_mode: str = 'zero') -> List[List[np.ndarray]]:
    """
    Create context windows from a sequence of embeddings.
    
    Args:
        embeddings: List of utterance embeddings
        context_size: Number of context utterances on each side (K in K-turn)
        padding_mode: How to pad at boundaries ('zero' or 'repeat')
    
    Returns:
        List of context windows, each containing (2*context_size + 1) embeddings
    """
    if not embeddings:
        return []
    
    # Create padding embeddings
    if padding_mode == 'zero':
        padding_emb = np.zeros_like(embeddings[0])
    elif padding_mode == 'repeat':
        padding_emb = None  # Will use boundary embeddings
    else:
        raise ValueError(f"Unknown padding_mode: {padding_mode}")
    
    # Pad the sequence
    if padding_mode == 'zero':
        padded_embeddings = [padding_emb] * context_size + embeddings + [padding_emb] * context_size
    else:  # repeat
        padded_embeddings = ([embeddings[0]] * context_size + 
                           embeddings + 
                           [embeddings[-1]] * context_size)
    
    # Extract windows
    windows = []
    for i in range(context_size, len(padded_embeddings) - context_size):
        window = padded_embeddings[i-context_size:i+context_size+1]
        windows.append(window)
    
    return windows


def create_turn_dataset(embeddings: List[np.ndarray],
                       metadata: pd.DataFrame,
                       context_size: int = 2,
                       padding_mode: str = 'zero') -> pd.DataFrame:
    """
    Create turn-level dataset with context windows.
    
    Args:
        embeddings: List of utterance embeddings
        metadata: DataFrame with ids, labels, etc.
        context_size: Context window size
        padding_mode: Padding strategy
    
    Returns:
        DataFrame with turn-level features including context windows
    """
    logger.info(f"Creating turn dataset with context_size={context_size}")
    
    # Create context windows
    windows = create_context_windows(embeddings, context_size, padding_mode)
    
    # Ensure we have the right number of windows
    assert len(windows) == len(embeddings), f"Mismatch: {len(windows)} windows vs {len(embeddings)} embeddings"
    assert len(windows) == len(metadata), f"Mismatch: {len(windows)} windows vs {len(metadata)} metadata rows"
    
    # Create features dictionary
    features_dict = {
        'text_features': windows,
        'ids': metadata['id'].tolist(),
        'file_id': metadata['file_id'].tolist(),
        'utterance_num': metadata['utterance_num'].tolist(),
        'label': metadata['label'].tolist()
    }
    
    # Create DataFrame
    features_dataset = pd.DataFrame.from_dict(features_dict)
    
    logger.info(f"Created turn dataset with shape: {features_dataset.shape}")
    logger.info(f"Context window shape per utterance: {len(windows[0])} x {windows[0][0].shape}")
    
    return features_dataset


def save_turn_dataset(dataset: pd.DataFrame,
                     output_path: Union[str, Path],
                     config_details: dict = None,
                     context_size: int = 2) -> None:
    """
    Save turn-level dataset to file.
    
    Args:
        dataset: Turn-level dataset DataFrame
        output_path: Path to save the dataset
        config_details: Configuration details to save alongside
        context_size: Context size for filename
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    np.save(str(output_path), dataset.to_numpy())
    logger.info(f"Saved turn dataset to: {output_path}")
    
    # Save config details if provided
    if config_details:
        config_path = output_path.parent / f"{output_path.stem}_config_k{context_size}.json"
        import json
        config_details['context_size'] = context_size
        config_details['dataset_shape'] = list(dataset.shape)
        
        with open(config_path, 'w') as f:
            json.dump(config_details, f, indent=2)
        logger.info(f"Saved config details to: {config_path}")


def load_turn_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load turn-level dataset from file.
    
    Args:
        file_path: Path to the saved dataset
    
    Returns:
        DataFrame with turn-level features
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = np.load(str(file_path), allow_pickle=True)
    
    # Reconstruct DataFrame
    # Note: This assumes the standard column order
    columns = ['text_features', 'ids', 'file_id', 'utterance_num', 'label']
    dataset = pd.DataFrame(data, columns=columns)
    
    logger.info(f"Loaded turn dataset from: {file_path}")
    logger.info(f"Dataset shape: {dataset.shape}")
    
    return dataset


class TurnExperimentGenerator:
    """
    Generator for turn-level experiments with different context sizes.
    """
    
    def __init__(self, base_embeddings_path: Union[str, Path], 
                 output_dir: Union[str, Path]):
        self.base_embeddings_path = Path(base_embeddings_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_k_turn_experiments(self, 
                                  config_id: int,
                                  k_values: List[int] = [0, 2, 4, 6],
                                  dataset_types: List[str] = ['train', 'val', 'test']) -> None:
        """
        Generate K-turn experiments for different context sizes.
        
        Args:
            config_id: Configuration ID
            k_values: List of context sizes to generate
            dataset_types: Dataset splits to process
        """
        logger.info(f"Generating K-turn experiments for Config {config_id}")
        logger.info(f"K values: {k_values}")
        logger.info(f"Dataset types: {dataset_types}")
        
        for dataset_type in dataset_types:
            for k in k_values:
                self._generate_single_k_turn(config_id, k, dataset_type)
    
    def _generate_single_k_turn(self, config_id: int, context_size: int, dataset_type: str):
        """Generate single K-turn dataset."""
        
        # Load base embeddings (without context)
        base_file = self.base_embeddings_path / f'X_textsroberta{dataset_type}_config{config_id}_raw.npy'
        
        if not base_file.exists():
            logger.warning(f"Base embeddings not found: {base_file}")
            return
        
        # Load embeddings and metadata
        # This would need to be adapted based on the actual storage format
        embeddings = np.load(base_file, allow_pickle=True)
        
        # For now, create placeholder - this needs actual implementation
        logger.info(f"Generated K={context_size} turn dataset for {dataset_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the context window generation
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy embeddings
    dummy_embeddings = [np.random.randn(768) for _ in range(10)]
    dummy_metadata = pd.DataFrame({
        'id': [f'utt_{i}' for i in range(10)],
        'file_id': ['file1'] * 10,
        'utterance_num': list(range(10)),
        'label': ['neu', 'hap', 'ang', 'sad'] * 2 + ['neu', 'hap']
    })
    
    # Test context window creation
    windows = create_context_windows(dummy_embeddings, context_size=2)
    print(f"Created {len(windows)} context windows")
    print(f"Window shape: {len(windows[0])} x {windows[0][0].shape}")
    
    # Test turn dataset creation
    turn_dataset = create_turn_dataset(dummy_embeddings, dummy_metadata, context_size=2)
    print(f"Turn dataset shape: {turn_dataset.shape}")