"""
WordNet-Affect Embeddings Generator
===================================
Generate emotion embeddings using WordNet-Affect and Word2Vec
"""

import os
import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim import downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
for resource in ['wordnet', 'stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource}")
        nltk.download(resource)

# Setup paths - Saturn Cloud structure 맞춤
HOME_DIR = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
EMBEDDINGS_DIR = HOME_DIR / 'data' / 'embeddings' / '4way' / 'wordnet'
WN_AFFECT_PATH = HOME_DIR / 'resources' / 'wn-affect-1.1' / 'a-synsets.xml'

# Create directories
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

stop_words = set(stopwords.words('english'))

def load_asynsets(corpus_path):
    """Load WordNet-Affect synsets from XML."""
    # ... (기존 함수 동일)

def find_similar_word(word, model, asynsets):
    """Find most similar word in WordNet-Affect."""
    # ... (기존 함수 동일)

def create_emotion_embeddings(asynsets, model):
    """Create emotion embeddings from WordNet-Affect."""
    # ... (기존 함수 동일)

def text_to_embedding(sentence, model, asynsets):
    """Convert text to WordNet-Affect based embedding."""
    # ... (기존 함수 개선 - error handling 추가)

def generate_and_save_embeddings(dataset_type, model_name, wn_affect_path):
    """Generate and save WordNet-Affect embeddings."""
    logger.info(f"Generating WordNet-Affect embeddings for {dataset_type} set")
    
    # Load data
    csv_path = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data_df = pd.read_csv(csv_path)
    
    # Filter out -1 labels
    valid_indices = data_df['label'] != '-1'
    data_df = data_df[valid_indices].reset_index(drop=True)
    
    logger.info(f"Loaded {len(data_df)} valid utterances from {dataset_type} set")
    
    # Load Word2Vec model (캐싱됨)
    logger.info(f"Loading {model_name} model...")
    model = api.load(model_name)
    
    # Load WordNet-Affect
    asynsets = load_asynsets(wn_affect_path)
    logger.info(f"Loaded WordNet-Affect with {sum(len(v) for v in asynsets.values())} synsets")
    
    # Generate embeddings
    embeddings = []
    for idx, sentence in enumerate(data_df['utterance']):
        if idx % 100 == 0:
            logger.info(f"Processing {idx}/{len(data_df)}...")
        
        embedding = text_to_embedding(sentence, model, asynsets)
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings)
    
    # Save embeddings
    output_file = EMBEDDINGS_DIR / f'{dataset_type}_embeddings.npy'
    np.save(output_file, embeddings_array)
    
    logger.info(f"Saved embeddings to {output_file}")
    logger.info(f"Shape: {embeddings_array.shape}")
    
    return embeddings_array

def main():
    parser = argparse.ArgumentParser(description="Generate WordNet-Affect based embeddings")
    parser.add_argument('--dataset', choices=['train', 'val', 'test', 'all'], 
                       default='all', help='Dataset split to process')
    parser.add_argument('--model', default='word2vec-google-news-300',
                       help='Word2Vec model name from gensim')
    parser.add_argument('--wn_affect_path', type=str, default=str(WN_AFFECT_PATH),
                       help='Path to WordNet-Affect XML file')
    
    args = parser.parse_args()
    
    # Check if WordNet-Affect file exists
    if not Path(args.wn_affect_path).exists():
        logger.error(f"WordNet-Affect file not found: {args.wn_affect_path}")
        logger.info("Please download WordNet-Affect from official source")
        sys.exit(1)
    
    # Determine datasets to process
    if args.dataset == 'all':
        datasets = ['train', 'val', 'test']
    else:
        datasets = [args.dataset]
    
    # Generate embeddings
    for dataset_type in datasets:
        try:
            embeddings = generate_and_save_embeddings(
                dataset_type, args.model, args.wn_affect_path
            )
            logger.info(f"✓ {dataset_type}: {embeddings.shape}")
        except Exception as e:
            logger.error(f"✗ Failed to generate {dataset_type}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("WordNet-Affect embedding generation completed!")

if __name__ == "__main__":
    main()


    
