"""
Modified baseline classifier training script focused on opensmile-sroberta embeddings.

Differences from the original:
- Only supports the opensmile-sroberta embedding configuration (expected dim 856)
- Accepts paths to the train/val/test .npy embedding files via command-line arguments
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             classification_report, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports (assumes classifiers module is on PYTHONPATH or in same package)
from classifiers import SimpleLSTM, MLP, FocalLoss, EarlyStopping, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory for CSV label files (unchanged)
HOME_DIR = Path("/home/jovyan/workspace/SenticCrystal")
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
RESULTS_DIR = HOME_DIR / 'results' / 'opensmile-sroberta'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Expected embedding dimension for opensmile-sroberta
EXPECTED_DIM = 856

def load_embeddings_and_labels(embedding_path: Path, dataset_type: str = 'train'):
    """Load embeddings from a provided .npy file and align with CSV labels.

    embedding_path: Path to .npy containing embeddings for all rows in the CSV order
    dataset_type: one of 'train', 'val', 'test' used to pick CSV file for labels
    Returns: embeddings_filtered, labels_encoded, label_encoder, df_filtered
    """
    logger.info(f"Loading embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")

    csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded CSV {csv_file} with {len(df)} rows")

    # Filter out -1 labels for training/evaluation
    valid_indices = df['label'] != '-1'
    df_filtered = df[valid_indices].reset_index(drop=True)

    # Validate embeddings length matches CSV rows; assume embeddings are in CSV order
    if embeddings.shape[0] != len(df):
        logger.warning(f"Embedding rows ({embeddings.shape[0]}) doesn't match CSV rows ({len(df)}). "
                       "Attempting to index by boolean mask if sizes allow.")

    try:
        embeddings_filtered = embeddings[valid_indices]
    except Exception:
        # Fallback: try to coerce embeddings to an ndarray and then index
        embeddings = np.asarray(embeddings)
        embeddings_filtered = embeddings[valid_indices]

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df_filtered['label'])

    logger.info(f"Valid samples (excluding -1): {len(df_filtered)}")
    logger.info(f"Filtered embeddings shape: {embeddings_filtered.shape}")
    return embeddings_filtered, labels_encoded, label_encoder, df_filtered


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Created data loaders with batch size: {batch_size}")
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, config, model_name="model"):
    device = torch.device(config['device'])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    best_val_acc = 0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1
    }


def evaluate_model(model, test_loader, label_encoder, device, model_name="model"):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names.tolist()
    }


def save_confusion_matrix_plot(cm, class_names, model_name, embedding_type, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - {embedding_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plot_file = save_dir / f'{embedding_type}_{model_name.lower()}_confusion_matrix.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot: {plot_file}")
    return plot_file


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier on opensmile-sroberta embeddings")
    parser.add_argument('--train_npy', type=Path, required=True, help='Path to train .npy embeddings')
    parser.add_argument('--val_npy', type=Path, required=True, help='Path to validation .npy embeddings')
    parser.add_argument('--test_npy', type=Path, required=True, help='Path to test .npy embeddings')
    parser.add_argument('--model', choices=['lstm', 'mlp', 'both'], default='both')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    args = parser.parse_args()

    # Load datasets using provided .npy paths
    X_train, y_train, label_encoder, train_df = load_embeddings_and_labels(args.train_npy, 'train')
    X_val, y_val, _, val_df = load_embeddings_and_labels(args.val_npy, 'val')
    X_test, y_test, _, test_df = load_embeddings_and_labels(args.test_npy, 'test')

    # Validate embedding dimension
    input_size = X_train.shape[-1]
    if input_size != EXPECTED_DIM:
        logger.warning(f"Warning: expected embedding dim {EXPECTED_DIM} but got {input_size}")

    num_classes = len(label_encoder.classes_)
    logger.info(f"Input size (embedding dimension): {input_size}")
    logger.info(f"Number of classes: {num_classes}")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size)

    # Training configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'device': DEFAULT_CONFIG.get('device', 'cpu')
    })

    results = {}
    models_to_train = ['lstm', 'mlp'] if args.model == 'both' else [args.model]

    for model_type in models_to_train:
        logger.info(f"Training {model_type} classifier")
        if model_type == 'lstm':
            model = SimpleLSTM(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes, dropout_rate=args.dropout_rate)
        else:
            model = MLP(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes, dropout_rate=args.dropout_rate)

        model_name = f"{model_type.upper()}-opensmile-sroberta"
        trained_model, training_history = train_model(model, train_loader, val_loader, config, model_name)

        device = torch.device(config['device'])
        test_results = evaluate_model(trained_model, test_loader, label_encoder, device, model_name)

        cm = np.array(test_results['confusion_matrix'])
        plot_file = save_confusion_matrix_plot(cm, test_results['class_names'], model_name, 'opensmile-sroberta', RESULTS_DIR)

        results[model_type] = {
            'embedding_type': 'opensmile-sroberta',
            'model_type': model_type,
            'embedding_dimension': input_size,
            'training_config': config,
            'training_history': training_history,
            'test_results': test_results,
            'plot_file': str(plot_file)
        }

        # Save per-model results
        model_results_file = RESULTS_DIR / f'opensmile_sroberta_{model_type}_baseline_results.json'
        with open(model_results_file, 'w') as f:
            json.dump(results[model_type], f, indent=2)
        logger.info(f"Results saved: {model_results_file}")

    # Save combined results
    combined_results_file = RESULTS_DIR / f'opensmile_sroberta_baseline_comparison.json'
    with open(combined_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Combined results saved: {combined_results_file}")

    logger.info("Baseline classification ablation completed!")


if __name__ == "__main__":
    main()
