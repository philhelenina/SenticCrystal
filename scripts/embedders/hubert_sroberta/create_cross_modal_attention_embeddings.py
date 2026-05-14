#!/usr/bin/env python3
"""
Create cross-modal attention embeddings in two directions:
1) q=text, k=v=audio  (text attends to audio)
2) q=audio, k=v=text  (audio attends to text)

Input defaults (IEMOCAP 4-way):
  Text : data/embeddings/4way/sroberta/avg_last4/wmean_pos_rev/{train,val,test}_filtered.npz
  Audio: data/embeddings/hubert_features/iemocap_{train,val,test}_hubert.npz

Output default:
  data/embeddings/4way/hubert_sroberta/cross_modal_attention/avg_last4/wmean_pos_rev/{train,val,test}.npz
  Output arrays inside each split file:
  - text_query_audio_kv_tokens
  - text_query_audio_kv
  - text_to_audio_attn_weights
  - audio_query_text_kv_tokens
  - audio_query_text_kv
  - audio_to_text_attn_weights

Notes:
- Supports both 2D arrays (N, D) and sequence arrays (N, T, D).
- For your current inputs, text and audio are effectively single-token sequences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _to_token_sequence(x: np.ndarray) -> np.ndarray:
    """Convert (N, D) or (N, 1, D) into canonical (N, T, D)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape={x.shape}")


def _load_text_embeddings(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    # Accept common keys: 'embeddings' or 'features', or a single array
    if "embeddings" in data:
        text = data["embeddings"]
    elif "features" in data:
        text = data["features"]
    elif len(data.files) == 1:
        text = data[data.files[0]]
    else:
        raise KeyError(f"No 'embeddings' or 'features' key in {npz_path}. Keys={data.files}")
    return _to_token_sequence(text)


def _load_audio_embeddings(audio_path: Path) -> np.ndarray:
    """Load audio embeddings from either .npy or .npz and return (N, T, D)."""
    if audio_path.suffix == ".npy":
        audio = np.load(audio_path, allow_pickle=True).astype(np.float32)
        if audio.ndim == 2:
            return audio[:, None, :]
        if audio.ndim == 3:
            return audio
        raise ValueError(f"Expected 2D or 3D audio array, got shape={audio.shape}")

    if audio_path.suffix == ".npz":
        data = np.load(audio_path, allow_pickle=True)
        if "features" in data:
            arr = data["features"]
        elif "embeddings" in data:
            arr = data["embeddings"]
        elif len(data.files) == 1:
            arr = data[data.files[0]]
        else:
            raise KeyError(f"No 'features' or 'embeddings' key in {audio_path}. Keys={data.files}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            return arr[:, None, :]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected 2D or 3D array in {audio_path}, got shape={arr.shape}")

    raise ValueError(f"Unsupported audio file extension: {audio_path}")


class BiDirectionalCrossModalAttention(nn.Module):
    """Cross-modal attention in both directions (text->audio and audio->text)."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # Direction 1: q=text, k=v=audio
        self.text_q_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_kv_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_to_audio_norm = nn.LayerNorm(hidden_dim)

        # Direction 2: q=audio, k=v=text
        self.audio_q_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_kv_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_to_text_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_tokens:  (N, T_text, text_dim)
            audio_tokens: (N, T_audio, audio_dim)

        Returns:
            Dict containing token-level outputs, pooled outputs, and attention weights
            for both directions.
        """
        # Case 1: q=text, k=v=audio
        q_text = self.text_q_proj(text_tokens)
        kv_audio = self.audio_kv_proj(audio_tokens)
        text_ctx, text_to_audio_w = self.text_to_audio_attn(
            query=q_text,
            key=kv_audio,
            value=kv_audio,
            need_weights=True,
            average_attn_weights=False,
        )
        text_ctx = self.text_to_audio_norm(text_ctx + q_text)
        text_ctx_pooled = text_ctx.mean(dim=1)

        # Case 2: q=audio, k=v=text
        q_audio = self.audio_q_proj(audio_tokens)
        kv_text = self.text_kv_proj(text_tokens)
        audio_ctx, audio_to_text_w = self.audio_to_text_attn(
            query=q_audio,
            key=kv_text,
            value=kv_text,
            need_weights=True,
            average_attn_weights=False,
        )
        audio_ctx = self.audio_to_text_norm(audio_ctx + q_audio)
        audio_ctx_pooled = audio_ctx.mean(dim=1)

        return {
            "text_query_audio_kv_tokens": text_ctx,
            "text_query_audio_kv": text_ctx_pooled,
            "text_to_audio_attn_weights": text_to_audio_w,
            "audio_query_text_kv_tokens": audio_ctx,
            "audio_query_text_kv": audio_ctx_pooled,
            "audio_to_text_attn_weights": audio_to_text_w,
        }


@torch.no_grad()
def run_split(
    model: BiDirectionalCrossModalAttention,
    text_path: Path,
    audio_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    text_np = _load_text_embeddings(text_path)
    audio_np = _load_audio_embeddings(audio_path)

    n = min(text_np.shape[0], audio_np.shape[0])
    if text_np.shape[0] != audio_np.shape[0]:
        print(
            f"  [warn] sample mismatch text={text_np.shape[0]} audio={audio_np.shape[0]} -> using n={n}"
        )
    text_np = text_np[:n]
    audio_np = audio_np[:n]

    text_t = torch.from_numpy(text_np).to(device)
    audio_t = torch.from_numpy(audio_np).to(device)
    outputs = model(text_t, audio_t)

    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bi-directional cross-modal attention embeddings")
    parser.add_argument(
        "--text-base",
        type=Path,
        default=Path("data/embeddings/4way/sroberta/avg_last4/wmean_pos_rev"),
        help="Base directory for text *_filtered.npz files",
    )
    parser.add_argument(
        "--audio-base",
        type=Path,
        default=Path("data/embeddings/hubert_features"),
        help="Base directory for audio HuBERT iemocap_*_hubert.npz files",
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("data/embeddings/4way/hubert_sroberta/cross_modal_attention/avg_last4/wmean_pos_rev"),
        help="Output directory for {split}.npz",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = ["train", "val", "test"]
    sample_text = _load_text_embeddings(args.text_base / "train_filtered.npz")
    sample_audio = _load_audio_embeddings(args.audio_base / "iemocap_train_hubert.npz")

    text_dim = sample_text.shape[-1]
    audio_dim = sample_audio.shape[-1]
    device = torch.device(args.device)

    print("=" * 72)
    print("Creating cross-modal attention embeddings")
    print("=" * 72)
    print(f"Text base : {args.text_base}")
    print(f"Audio base: {args.audio_base}")
    print(f"Out base  : {args.out_base}")
    print(f"Dims      : text={text_dim}, audio={audio_dim}, hidden={args.hidden_dim}")
    print(f"Heads     : {args.num_heads}")
    print(f"Device    : {device}")

    model = BiDirectionalCrossModalAttention(
        text_dim=text_dim,
        audio_dim=audio_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    model.eval()

    args.out_base.mkdir(parents=True, exist_ok=True)
    for split in splits:
        print(f"\n[split={split}]")
        text_path = args.text_base / f"{split}_filtered.npz"
        # HuBERT files are named like: iemocap_train_hubert.npz
        audio_path = args.audio_base / f"iemocap_{split}_hubert.npz"
        split_outputs = run_split(model, text_path, audio_path, device)

        out_path = args.out_base / f"{split}.npz"
        np.savez_compressed(out_path, **split_outputs)
        print(
            f"  saved: {out_path} | "
            f"text->audio={split_outputs['text_query_audio_kv'].shape}, "
            f"audio->text={split_outputs['audio_query_text_kv'].shape}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
