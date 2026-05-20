"""
Train baseline classifiers on cross-modal attention embeddings.

This script mirrors the baseline ablation workflow but consumes embeddings created by:
scripts/embedders/opensmile_sroberta/create_cross_modal_attention_embeddings.py

Supported feature modes:
- text_query_audio_kv
- audio_query_text_kv
- concat_pooled (concatenate both pooled directions)
- text_query_audio_kv_tokens
- audio_query_text_kv_tokens

Running:
python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --model both
python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --feature_mode text_query_audio_kv --model both
python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --feature_mode audio_query_text_kv --model both

"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from classifiers import DEFAULT_CONFIG, EarlyStopping, MLP, SimpleLSTM


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HOME_DIR = Path("./")
DATA_DIR = HOME_DIR / "data" / "iemocap_4way_data"
DEFAULT_EMBEDDING_DIR = (
    HOME_DIR
    / "data"
    / "embeddings"
    / "4way"
    / "opensmile_sroberta"
    / "cross_modal_attention"
    / "avg_last4"
    / "wmean_pos_rev"
)
DEFAULT_RESULTS_DIR = (
    HOME_DIR
    / "results"
    / "4way"
    / "opensmile-sroberta"
    / "cross_modal_attention"
)


def load_split_features_and_labels(
    embedding_dir: Path,
    dataset_type: str,
    feature_mode: str,
    token_pool: str,
):
    """Load one split and align features with non -1 labels from CSV."""
    csv_file = DATA_DIR / f"{dataset_type}_filtered.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    npz_file = embedding_dir / f"{dataset_type}.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {npz_file}")

    df = pd.read_csv(csv_file)
    valid_mask = (df["label"] != "-1").to_numpy()
    labels_filtered = df.loc[valid_mask, "label"].to_numpy()

    data = np.load(npz_file, allow_pickle=True)

    if feature_mode == "concat_pooled":
        k1 = "text_query_audio_kv"
        k2 = "audio_query_text_kv"
        if k1 not in data or k2 not in data:
            raise KeyError(f"Missing keys for concat_pooled in {npz_file}. Keys={data.files}")
        features = np.concatenate([data[k1], data[k2]], axis=-1)
    else:
        if feature_mode not in data:
            raise KeyError(f"Feature key '{feature_mode}' not found in {npz_file}. Keys={data.files}")
        features = data[feature_mode]

    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 3 and token_pool != "keep":
        if token_pool == "mean":
            features = features.mean(axis=1)
        elif token_pool == "flatten":
            features = features.reshape(features.shape[0], -1)
        else:
            raise ValueError(f"Unsupported token_pool: {token_pool}")

    if features.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D features, got shape={features.shape}")

    # Align to filtered labels using the most plausible mapping strategy.
    if features.shape[0] == len(df):
        features_filtered = features[valid_mask]
    elif features.shape[0] == valid_mask.sum():
        features_filtered = features
    else:
        n = min(features.shape[0], valid_mask.sum())
        logger.warning(
            "Split %s size mismatch (features=%d, valid_labels=%d). Truncating to n=%d",
            dataset_type,
            features.shape[0],
            int(valid_mask.sum()),
            n,
        )
        features_filtered = features[:n]
        labels_filtered = labels_filtered[:n]

    logger.info(
        "Loaded %s | feature_mode=%s | token_pool=%s | shape=%s | labels=%d",
        dataset_type,
        feature_mode,
        token_pool,
        features_filtered.shape,
        len(labels_filtered),
    )

    return features_filtered, labels_filtered


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, config, model_name):
    device = torch.device(config["device"])
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    logger.info("Training %s", model_name)
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        val_acc = correct / max(1, total)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)
        if (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1,
                config["num_epochs"],
                train_loss,
                val_loss,
                val_acc,
            )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_acc": best_val_acc,
        "final_epoch": epoch + 1,
    }


def evaluate_model(model, test_loader, label_encoder, device, model_name):
    logger.info("Evaluating %s", model_name)

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_predictions)
    class_names = label_encoder.classes_
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_names": class_names.tolist(),
    }


def save_confusion_matrix_plot(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def adapt_features_for_model(X, model_type):
    """Ensure feature shape is compatible with selected classifier."""
    if model_type == "mlp" and X.ndim == 3:
        # MLP consumes 2D vectors; mean pooling over tokens keeps dimensionality stable.
        return X.mean(axis=1)
    return X


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline classifier(s) on cross-modal attention embeddings"
    )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing {train,val,test}.npz cross-modal files",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="concat_pooled",
        choices=[
            "text_query_audio_kv",
            "audio_query_text_kv",
            "concat_pooled",
            "text_query_audio_kv_tokens",
            "audio_query_text_kv_tokens",
        ],
        help="Which feature array from split .npz to train on",
    )
    parser.add_argument(
        "--token_pool",
        type=str,
        default="mean",
        choices=["mean", "flatten", "keep"],
        help="How to convert token-level (3D) embeddings before training",
    )
    parser.add_argument("--model", choices=["lstm", "mlp", "both"], default="both")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base output directory for metrics and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"{args.feature_mode}_{args.token_pool}"
    results_dir = args.results_dir / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Embedding dir: %s", args.embedding_dir)
    logger.info("Results dir: %s", results_dir)
    logger.info("Feature mode: %s | token_pool=%s", args.feature_mode, args.token_pool)

    X_train, y_train_raw = load_split_features_and_labels(
        args.embedding_dir, "train", args.feature_mode, args.token_pool
    )
    X_val, y_val_raw = load_split_features_and_labels(
        args.embedding_dir, "val", args.feature_mode, args.token_pool
    )
    X_test, y_test_raw = load_split_features_and_labels(
        args.embedding_dir, "test", args.feature_mode, args.token_pool
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    num_classes = len(label_encoder.classes_)
    logger.info("Classes: %s", label_encoder.classes_)

    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
        }
    )

    models_to_train = ["lstm", "mlp"] if args.model == "both" else [args.model]
    results = {}

    for model_type in models_to_train:
        logger.info("%s", "=" * 70)
        logger.info("Training %s on %s", model_type.upper(), run_name)

        X_train_model = adapt_features_for_model(X_train, model_type)
        X_val_model = adapt_features_for_model(X_val, model_type)
        X_test_model = adapt_features_for_model(X_test, model_type)

        input_size = X_train_model.shape[-1]
        logger.info("Model input shape (train): %s", X_train_model.shape)

        train_loader, val_loader, test_loader = create_data_loaders(
            X_train_model,
            y_train,
            X_val_model,
            y_val,
            X_test_model,
            y_test,
            args.batch_size,
        )

        if model_type == "lstm":
            model = SimpleLSTM(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_classes=num_classes,
                dropout_rate=args.dropout_rate,
            )
        else:
            model = MLP(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_classes=num_classes,
                dropout_rate=args.dropout_rate,
            )

        model_name = f"{model_type.upper()}-cross-modal-{run_name}"
        trained_model, training_history = train_model(
            model,
            train_loader,
            val_loader,
            config,
            model_name,
        )

        device = torch.device(config["device"])
        test_results = evaluate_model(
            trained_model,
            test_loader,
            label_encoder,
            device,
            model_name,
        )

        cm = np.array(test_results["confusion_matrix"])
        cm_file = results_dir / f"{model_type}_{run_name}_confusion_matrix.png"
        save_confusion_matrix_plot(
            cm,
            test_results["class_names"],
            f"{model_name} Confusion Matrix",
            cm_file,
        )
        logger.info("Saved confusion matrix: %s", cm_file)

        result_payload = {
            "embedding_source": str(args.embedding_dir),
            "feature_mode": args.feature_mode,
            "token_pool": args.token_pool,
            "model_type": model_type,
            "embedding_dimension": int(input_size),
            "model_params": {
                "input_size": int(input_size),
                "hidden_size": int(args.hidden_size),
                "num_classes": int(num_classes),
            },
            "training_config": config,
            "training_history": training_history,
            "test_results": test_results,
            "plot_file": str(cm_file),
        }
        results[model_type] = result_payload

        model_json = results_dir / f"{model_type}_{run_name}_results.json"
        with open(model_json, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        logger.info("Saved model results: %s", model_json)

    combined_json = results_dir / f"comparison_{run_name}.json"
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved combined results: %s", combined_json)

    logger.info("%s", "=" * 70)
    logger.info("Cross-modal baseline summary")
    logger.info("Feature mode: %s | token_pool=%s", args.feature_mode, args.token_pool)
    for model_type in models_to_train:
        test = results[model_type]["test_results"]
        logger.info(
            "%s: Acc=%.4f Macro-F1=%.4f Weighted-F1=%.4f",
            model_type.upper(),
            test["accuracy"],
            test["macro_f1"],
            test["weighted_f1"],
        )


if __name__ == "__main__":
    main()

# Example usages:
# python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --model both
# python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --feature_mode text_query_audio_kv --model lstm
# python3 scripts/models/opensmile_sroberta/train_cross_modal_attention_classifier.py --feature_mode audio_query_text_kv_tokens --token_pool mean --model mlp