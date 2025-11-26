#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sroberta_hier_npz.py
- 발화(여러 문장) → 문장별 임베딩(토큰 풀링) → (N, S_max, 768) + lengths
- 4way/6way 자동 감지 (data_dir 기준)
- 레이어: last / avg_last4 / last4_scalar / last4_scalar_up/down/top2
- 토큰풀링: cls / mean / attn(τ) / wmean_pos / wmean_pos_rev / 
           wmean_exp_fast / wmean_exp_med / wmean_exp_slow / wmean_idf
- 저장: {OUT_ROOT}/sentence-roberta-hier/{layer}/{pool}/{split}.npz
"""
import argparse, math, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
IDF_DEFAULT = HOME / "src" / "features" / "idf.csv"

PRESETS = {
    "last4_scalar":       [1,1,1,1],
    "last4_scalar_up":    [1,2,3,4],   # 오래된→최근 레이어로 가중 증가
    "last4_scalar_down":  [4,3,2,1],
    "last4_scalar_top2":  [0,1,1,0],   # 상위 2개 강조
}

def detect_task(data_dir: Path) -> str:
    """4way or 6way 자동 감지"""
    if "4way" in str(data_dir):
        return "4way"
    elif "6way" in str(data_dir):
        return "6way"
    else:
        raise ValueError(f"Cannot detect task from {data_dir}")

def simple_sent_split(text: str) -> List[str]:
    # 매우 보수적인 문장 분리기(., !, ? + 공백) — 공백/쉼표 앞뒤 트리밍
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float()
    s = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (last_hidden * m.unsqueeze(-1)).sum(dim=1) / s

def attn_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    cls = last_hidden[:, 0, :]
    scores = (last_hidden * cls.unsqueeze(1)).sum(dim=-1) / math.sqrt(last_hidden.size(-1))
    scores = scores.masked_fill(attn_mask == 0, -1e9)
    w = torch.softmax(scores / max(tau, 1e-6), dim=1)
    return (last_hidden * w.unsqueeze(-1)).sum(dim=1)

def pos_weights(
    attn_mask: torch.Tensor,
    mode: str = "linear_rev",  # "linear_rev", "linear", "exp_decay"
    tau: float = 5.0
) -> torch.Tensor:
    """
    위치 기반 가중치 생성
    
    Args:
        attn_mask: (B, T) attention mask
        mode: "linear_rev" (T..1), "linear" (1..T), "exp_decay" (e^(-(t-1)/τ))
        tau: exponential decay constant (작을수록 빠른 붕괴)
        
    Returns:
        w: (B, T) normalized weights
    """
    B, T = attn_mask.shape
    device = attn_mask.device
    
    if mode == "linear_rev":
        # T..1 (문말 가중↑)
        pos = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.flip(pos, dims=[0])
        
    elif mode == "linear":
        # 1..T (문두 가중↑)
        w_raw = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        
    elif mode == "exp_decay":
        # Exponential decay: w_t = e^(-(t-1)/τ)
        # t=1에서 최댓값 1, t 증가하면 지수 감소
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.exp(-(t - 1) / tau)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Apply mask and normalize
    w = w_raw.unsqueeze(0).expand(B, T) * attn_mask.float()
    s = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return w / s

def idf_weights(token_ids: torch.Tensor, attn_mask: torch.Tensor, tokenizer, idf_map: Dict[str, float]) -> torch.Tensor:
    B, T = token_ids.shape
    weights = torch.ones((B, T), dtype=torch.float32, device=token_ids.device)
    # 안전하게 convert_ids_to_tokens를 per-row로 사용
    for b in range(B):
        ids = token_ids[b].tolist()
        msk = attn_mask[b].tolist()
        row=[]
        for tid, m in zip(ids, msk):
            if m==0:
                row.append(0.0); continue
            t = tokenizer.convert_ids_to_tokens([tid])[0]
            t = t.replace("Ġ","").lower()
            row.append(1.0 + float(idf_map.get(t, 0.0)))
        weights[b] = torch.tensor(row, device=token_ids.device)
    s = (weights * attn_mask.float()).sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (weights * attn_mask.float()) / s

def combine_layers(hidden_states: Tuple[torch.Tensor, ...], mode: str, scalar_w: Optional[List[float]]):
    if mode == "last":
        return hidden_states[-1]
    if mode == "avg_last4":
        return torch.stack(hidden_states[-4:], dim=0).mean(0)
    if mode.startswith("last4_scalar"):
        if scalar_w is None:
            raise ValueError("scalar_w required for last4_scalar*")
        w = torch.tensor(scalar_w, dtype=torch.float32, device=hidden_states[-1].device)
        w = w / (w.sum() + 1e-8)
        hs = torch.stack(hidden_states[-4:], dim=0)
        return (hs * w.view(4,1,1,1)).sum(0)
    raise ValueError(f"unknown layer mode: {mode}")

def find_text_column(df: pd.DataFrame) -> str:
    cands = ["text","utterance","utt","sentence","transcript"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype == object: return c
    raise ValueError(f"text column not found. cols={df.columns[:10].tolist()}")

def build_parser():
    ap = argparse.ArgumentParser(description="Generate Sentence-RoBERTa hierarchical embeddings (4way/6way auto-detect)")
    ap.add_argument("--model_name", default="sentence-transformers/nli-roberta-base-v2")
    ap.add_argument("--data_dir", type=str, required=True,
                   help="e.g., data/iemocap_4way_data or data/iemocap_6way_data")
    ap.add_argument("--out_root", type=str, required=True,
                   help="e.g., data/embeddings/4way/sentence-roberta-hier")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--text_col", default=None)
    
    # 레이어
    ap.add_argument("--layer", 
                   choices=["last","avg_last4","last4_scalar",
                           "last4_scalar_up","last4_scalar_down","last4_scalar_top2"], 
                   default="avg_last4")
    ap.add_argument("--scalar_weights", default="")  # last4_scalar 직접 지정시 "a,b,c,d"
    
    # 토큰 풀링
    ap.add_argument("--poolings", nargs="+", default=["mean"], 
                    choices=["cls","mean","attn",
                            "wmean_pos","wmean_pos_rev",
                            "wmean_exp_fast","wmean_exp_med","wmean_exp_slow",
                            "wmean_idf"])
    ap.add_argument("--attn_tau", type=float, default=1.0)
    
    # Exponential decay constants
    ap.add_argument("--exp_tau_fast", type=float, default=2.0,
                   help="τ for fast decay (high-arousal emotions)")
    ap.add_argument("--exp_tau_med", type=float, default=5.0,
                   help="τ for medium decay")
    ap.add_argument("--exp_tau_slow", type=float, default=10.0,
                   help="τ for slow decay (low-arousal emotions)")
    
    ap.add_argument("--idf_csv", default=str(IDF_DEFAULT))
    
    # 기타
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)  # 문장 단위 배치
    ap.add_argument("--cpu", action="store_true")
    return ap

def main():
    args = build_parser().parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")
    
    data_dir = Path(args.data_dir)
    task = detect_task(data_dir)
    print(f"[INFO] Detected task: {task}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)
    print(f"[INFO] Model loaded: {args.model_name}")

    # 레이어 스칼라 가중
    layer_mode = args.layer
    scalar_w = None
    if layer_mode.startswith("last4_scalar"):
        if args.scalar_weights:
            scalar_w = [float(x) for x in args.scalar_weights.split(",")]
        else:
            scalar_w = PRESETS[layer_mode]
            layer_mode = "last4_scalar"
        print(f"[INFO] Layer scalar weights: {scalar_w}")

    # IDF 로드 필요시
    idf_map = None
    if any("idf" in p for p in args.poolings):
        idf_path = Path(args.idf_csv)
        if idf_path.exists():
            df_idf = pd.read_csv(idf_path)
            idf_map = {str(t).lower(): float(i) for t,i in zip(df_idf["token"], df_idf["idf"])}
            print(f"[INFO] IDF map loaded: {len(idf_map)} entries")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Pooling methods: {args.poolings}")
    print(f"[INFO] Exponential τ: fast={args.exp_tau_fast}, med={args.exp_tau_med}, slow={args.exp_tau_slow}")

    # Process each split
    for split in args.splits:
        csv_path = data_dir / f"{split}_{task}_unified.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        col = args.text_col or find_text_column(df)
        texts = df[col].astype(str).fillna("").tolist()
        print(f"\n[{split}] Processing {len(texts)} utterances from {csv_path}")

        per_pool_arrays = {p: [] for p in args.poolings}
        lengths = []

        for idx, txt in enumerate(texts):
            sents = simple_sent_split(txt)
            if not sents: 
                sents = [txt.strip()]
            lengths.append(len(sents))

            # 문장 묶음을 한 번에 배치 인코딩
            enc = tokenizer(sents, padding=True, truncation=True, 
                          max_length=args.max_length, return_tensors="pt")
            enc = to_device(enc, device)
            
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
                last_h = combine_layers(out.hidden_states, layer_mode, scalar_w)  # (B_s, T, H)
                attn_mask = enc["attention_mask"]

                # 문장별 토큰풀링 → (B_s, H)
                pool_outs = {}
                for p in args.poolings:
                    if p=="cls":
                        pooled = last_h[:,0,:]
                    elif p=="mean":
                        pooled = masked_mean(last_h, attn_mask)
                    elif p=="attn":
                        pooled = attn_pool(last_h, attn_mask, tau=args.attn_tau)
                    elif p=="wmean_pos":
                        w = pos_weights(attn_mask, mode="linear")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p=="wmean_pos_rev":
                        w = pos_weights(attn_mask, mode="linear_rev")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    
                    # Exponential decay variants
                    elif p=="wmean_exp_fast":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_fast)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p=="wmean_exp_med":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_med)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p=="wmean_exp_slow":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_slow)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    
                    elif p=="wmean_idf":
                        w = idf_weights(enc["input_ids"], attn_mask, tokenizer, idf_map)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    else:
                        raise ValueError(p)
                    pool_outs[p] = pooled.detach().cpu().numpy().astype(np.float32)  # (S_i, 768)

                for p in args.poolings:
                    per_pool_arrays[p].append(pool_outs[p])

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(texts)} utterances...")

        # 패딩 후 저장
        N = len(texts)
        S_max = max(lengths) if lengths else 1
        for p in args.poolings:
            H = model.config.hidden_size
            out = np.zeros((N, S_max, H), dtype=np.float32)
            for i, mat in enumerate(per_pool_arrays[p]):
                s = mat.shape[0]
                out[i, :s, :] = mat
            
            save_dir = out_root / layer_mode / p
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{split}.npz"
            
            np.savez_compressed(out_path,
                                embeddings=out, 
                                lengths=np.asarray(lengths, dtype=np.int32))
            print(f"  [SAVE] {layer_mode}/{p}/{split} → {out_path}  shape={out.shape}  (S_max={S_max})")

    print("\n[DONE] All splits saved.")

if __name__ == "__main__":
    main()
