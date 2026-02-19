"""
Universal Classifier Training Script
===================================

Train LSTM or MLP classifiers with embeddings generated from any configuration.
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.classifiers import SimpleLSTM, MLP, FocalLoss, EarlyStopping, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
EMBEDDINGS_DIR = HOME_DIR / 'scripts' / 'embeddings'
RESULTS_DIR = HOME_DIR / 'results' / 'classifiers'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings_and_labels(config_id, dataset_type='train'):
    """Load embeddings and corresponding labels."""
    
    # Load embeddings
    embedding_file = EMBEDDINGS_DIR / f'X_textsroberta{dataset_type}_config{config_id}.npy'
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    
    embeddings = np.load(embedding_file, allow_pickle=True)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Load labels from CSV
    csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Filter out -1 labels
    valid_indices = df['label'] != '-1'
    df_filtered = df[valid_indices].reset_index(drop=True)
    
    # Extract text features from embeddings (assuming context window format)
    if embeddings.ndim == 1:
        # Convert from object array to proper format
        embeddings_array = []
        for i, row in enumerate(embeddings):
            if valid_indices.iloc[i]:
                # Extract middle element from context window (assuming 5-element context)
                text_features = row[2] if hasattr(row, '__len__') and len(row) > 2 else row
                embeddings_array.append(text_features)
        embeddings_filtered = np.array(embeddings_array)
    else:
        embeddings_filtered = embeddings[valid_indices]
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df_filtered['label'])
    
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Valid samples (excluding -1): {len(df_filtered)}")
    logger.info(f"Label distribution: {df_filtered['label'].value_counts().to_dict()}")
    logger.info(f"Final embeddings shape: {embeddings_filtered.shape}")
    
    return embeddings_filtered, labels_encoded, label_encoder, df_filtered['label']


def train_classifier(model, train_loader, val_loader, config):
    """Train the classifier with early stopping."""
    
    device = torch.device(config['device'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # Training
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
        
        # Validation
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
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}] '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.4f}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, test_loader, label_encoder, device):
    """Evaluate model on test set."""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    macro_f1 = precision_recall_fscore_support(all_labels, all_predictions, average='macro')[2]
    weighted_f1 = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[2]
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    logger.info(f"\nTest Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, digits=4)
    logger.info(f"\nClassification Report:\n{report}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report
    }


def main():
    parser = argparse.ArgumentParser(description="Train classifier on generated embeddings")
    parser.add_argument('--config_id', type=int, required=True, help='Configuration ID for embeddings')
    parser.add_argument('--model', choices=['lstm', 'mlp'], required=True, help='Model type')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading embeddings for Config {args.config_id}")
    X_train, y_train, label_encoder, _ = load_embeddings_and_labels(args.config_id, 'train')
    X_val, y_val, _, _ = load_embeddings_and_labels(args.config_id, 'val') 
    X_test, y_test, _, _ = load_embeddings_and_labels(args.config_id, 'test')
    
    # Convert to tensors and create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[-1]
    num_classes = len(label_encoder.classes_)
    
    if args.model == 'lstm':
        model = SimpleLSTM(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes)
    else:
        model = MLP(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes)
    
    logger.info(f"Created {args.model.upper()} model: {model}")
    
    # Training config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs
    })
    
    # Train model
    logger.info("Starting training...")
    trained_model, training_history = train_classifier(model, train_loader, val_loader, config)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    device = torch.device(config['device'])
    test_results = evaluate_model(trained_model, test_loader, label_encoder, device)
    
    # Save results
    save_path = RESULTS_DIR / f'config{args.config_id}_{args.model}_results.json'
    results = {
        'config_id': args.config_id,
        'model_type': args.model,
        'training_config': config,
        'training_history': training_history,
        'test_results': test_results
    }
    
    import json
    with open(save_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results['test_results'].items():
            if isinstance(value, np.ndarray):
                results['test_results'][key] = value.tolist()
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()