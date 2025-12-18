#!/usr/bin/env python3
# emoroberta_embeddings_intel.py
# - 인텔 맥용 수정 버전
# - 원본 코드에서 문제가 될 수 있는 부분만 수정

import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

# ===== Safe header =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Safe header loaded. Starting script...")
sys.stdout.flush()

# Paths
HOME_DIR = Path("/Users/helenjeong/Projects/SenticCrystal")
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
EMBEDDINGS_DIR = HOME_DIR / 'scripts' / 'embeddings' / '4way' / '4way-emo-roberta'

# ===== Settings =====
MODEL_NAME_OR_PATH = "/Users/helenjeong/Projects/SenticCrystal/saturn_cloud_deployment/src/features/emoroberta-local"
MAX_LENGTH = 256
BATCH_SIZE = 32
POOLING = "mean"  # choices: "cls", "mean"
SELECT_LAYER = -1  # -1: last hidden state

# ===== TF / HF imports & device prep =====
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TensorFlow imported")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading tokenizer/model from: {MODEL_NAME_OR_PATH}")
sys.stdout.flush()

# Load tokenizer with safe settings
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    use_fast=False,
    local_files_only=True
)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tokenizer loaded.")
sys.stdout.flush()

# Load model
model = TFAutoModel.from_pretrained(
    MODEL_NAME_OR_PATH,
    output_hidden_states=True,
    local_files_only=True
)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded.")
sys.stdout.flush()

# ===== Utilities =====
def load_texts(split: str) -> list[str]:
    csv_path = DATA_DIR / f"{split}_4way_with_minus_one.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "utterance" not in df.columns:
        raise KeyError(f"'utterance' column not found in {csv_path.name}")
    # 임베딩은 -1 포함 전체 행 생성 (나중에 분류 시에만 -1 필터)
    texts = df["utterance"].fillna("").astype(str).tolist()
    print(f"[{split}] rows={len(texts)} | csv={csv_path.name}")
    return texts

def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H] (tf.Tensor), mask: [B, T]
    mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), dtype=last_hidden_state.dtype)
    masked = last_hidden_state * mask
    sum_embeddings = tf.reduce_sum(masked, axis=1)
    lengths = tf.reduce_sum(mask, axis=1) + 1e-9
    return sum_embeddings / lengths

def embed_texts(texts: list[str]) -> np.ndarray:
    n = len(texts)
    outs = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="tf"
        )
        # forward
        out = model(**enc)
        # pick layer
        hidden_states = out.hidden_states  # tuple(len=L+1): [B,T,H]
        layer_out = hidden_states[SELECT_LAYER]  # last layer by default

        if POOLING == "cls":
            # CLS = first token
            cls = layer_out[:, 0, :]  # [B, H]
            pooled = cls
        elif POOLING == "mean":
            pooled = mean_pool(layer_out, enc["attention_mask"])  # [B, H]
        else:
            raise ValueError(f"Unsupported POOLING={POOLING}")

        outs.append(pooled.numpy())
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Encoded {min(i+BATCH_SIZE, n)}/{n}")
            sys.stdout.flush()
    return np.concatenate(outs, axis=0)

def save_embeddings(split: str, X: np.ndarray):
    out_dir = EMBEDDINGS_DIR / POOLING
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_embeddings.npy"
    np.save(out_path, X)
    print(f"[OK] Saved: {out_path}  shape={X.shape}")

# ===== Main run (all three splits) =====
def main():
    meta = {"model": MODEL_NAME_OR_PATH, "pooling": POOLING, "layer": SELECT_LAYER,
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE}

    for split in ["train", "val", "test"]:
        texts = load_texts(split)
        X = embed_texts(texts)
        save_embeddings(split, X)
        meta[f"{split}_rows"] = len(texts)
        meta[f"{split}_shape"] = list(X.shape)

    # meta json
    meta_path = EMBEDDINGS_DIR / POOLING / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved meta: {meta_path}")

if __name__ == "__main__":
    main()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done.")