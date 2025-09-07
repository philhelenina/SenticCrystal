# filename changed from word_sentence_utterance_all_val.py to embeddings.py (Jan 27, 2025)
# file paths changed
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gensim.downloader as api
import torch
import json
import argparse
from wnaffect_module import WordNetAffectEmbedder
from sroberta_module import SentenceEmbedder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path("/Volumes/ssd/01-ckj-postdoc/Lingua-Emoca/")
logging.info(f"Home Directory is: {HOME_DIR}")

word2vec_model = api.load('word2vec-google-news-300')
wn_affect_path = HOME_DIR / 'scripts' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-synsets.xml'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentence_embedder = SentenceEmbedder("nli-distilroberta-base-v2", device)
wn_embedder = WordNetAffectEmbedder(word2vec_model, wn_affect_path)

def get_paths(dataset_type):
    """Set file paths based on dataset type (train/val/test)."""
    csv_path = HOME_DIR / 'data' / 'preprocessed-iemocap' / '4way' / f'{dataset_type}_4way_with_minus_one.csv'
    dataset_path = HOME_DIR / 'scripts'/ 'embeddings'
    return csv_path, dataset_path

def load_configurations(file_path='configurations.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {config['id']: config for config in data['configurations']}

def print_configurations(configurations):
    for config_id, config in configurations.items():
        print(f"\nConfiguration {config_id}:")
        for key, value in config.items():
            if key != 'id':
                print(f"  {key}: {value}")

def generate_and_save_embeddings(config_id, config, sentence_embedder, wn_embedder, csv_path, dataset_type):
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} utterances from {csv_path}")

    utterances = df['utterance'].tolist()
    final_ids = df['id'].tolist()
    file_ids = df['file_id'].tolist()
    utterance_nums = df['utterance_num'].tolist()
    labels = df['label'].tolist()

    logger.info(f"Number of utterances to process: {len(utterances)}")

    alpha_range = config.get('alpha_range', [None])

    dataset_path = HOME_DIR / 'scripts'/ 'embeddings'
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    for alpha in alpha_range:
        logger.info(f"Processing with alpha: {alpha}")
        embeddings = []

        for i, utterance in enumerate(utterances):
            if i % 100 == 0:
                logger.info(f"Processing utterance {i+1}/{len(utterances)}")
            try:
                embedding = sentence_embedder.process_utterance(
                    utterance,
                    wn_embedder,
                    apply_word_pe=config["apply_word_pe"],
                    pooling_method=config["pooling_method"],
                    apply_sentence_pe=config["apply_sentence_pe"],
                    combination_method=config["combination_method"],
                    alpha=alpha
                )
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error processing utterance {i+1}: {str(e)}")
                embeddings.append(np.zeros(768))

        context_size = 2
        padded_embeddings = [np.zeros_like(embeddings[0])] * context_size + embeddings + [np.zeros_like(embeddings[0])] * context_size

        features_dict = {
            'text_features': [padded_embeddings[i-context_size:i+context_size+1] for i in range(context_size, len(padded_embeddings)-context_size)],
            'ids': final_ids,
            'file_id': file_ids,
            'utterance_num': utterance_nums,
            'label': labels
        }
        features_dataset = pd.DataFrame.from_dict(features_dict)

        alpha_str = f"_alpha{alpha}" if alpha is not None else ""
        file_name = f'X_textsroberta{dataset_type}_config{config_id}{alpha_str}.npy'
        np.save(str(dataset_path / file_name), features_dataset.to_numpy())

        logger.info(f"Saved embeddings for configuration {config_id}, alpha={alpha}. Shape: {features_dataset.shape}")

    with open(dataset_path / f"config{config_id}details{dataset_type}.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Finished generating and saving embeddings.")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a specific configuration")
    parser.add_argument("--list", action="store_true", help="List all available configurations")
    parser.add_argument("--config", type=int, help="ID of the configuration to use")
    parser.add_argument("--train", action="store_true", help="Generate embeddings for training data")
    parser.add_argument("--val", action="store_true", help="Generate embeddings for validation data")
    parser.add_argument("--test", action="store_true", help="Generate embeddings for test data")

    args = parser.parse_args()

    logger.info("Loading configurations...")
    configurations = load_configurations()
    logger.info(f"Loaded {len(configurations)} configurations.")

    if args.list:
        logger.info("Listing configurations...")
        print_configurations(configurations)
        return

    if args.config is None:
        print_configurations(configurations)
        config_id = int(input("Enter the ID of the configuration you want to use: "))
    else:
        config_id = args.config

    logger.info(f"Selected configuration ID: {config_id}")

    if config_id not in configurations:
        logger.error(f"Invalid configuration ID. Must be one of {list(configurations.keys())}")
        return

    config = configurations[config_id]

    if args.train:
        logger.info("Generating embeddings for training data...")
        csv_path, _ = get_paths('train')
        generate_and_save_embeddings(config_id, config, sentence_embedder, wn_embedder, csv_path, 'train')

    if args.val:
        logger.info("Generating embeddings for validation data...")
        csv_path, _ = get_paths('val')
        generate_and_save_embeddings(config_id, config, sentence_embedder, wn_embedder, csv_path, 'val')

    if args.test:
        logger.info("Generating embeddings for test data...")
        csv_path, _ = get_paths('test')
        generate_and_save_embeddings(config_id, config, sentence_embedder, wn_embedder, csv_path, 'test')

    if not (args.train or args.val or args.test):
        logger.warning("Please specify --train, --val, --test, or any combination of them.")

    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()

# python embeddings.py --config CONFIG_ID --train --val --test
