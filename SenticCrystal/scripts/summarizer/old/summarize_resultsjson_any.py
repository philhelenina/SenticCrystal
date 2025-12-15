#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_resultsjson_any.py
- Recursively scan given roots for results.json (any model type)
- Aggregate by (embedding_type, variant/layer/pool, model) across seeds
- Save rows CSV + summary CSV, and print TOP-K tables
"""
from pathlib import Path
import argparse, json, re
import pandas as pd
import numpy as np

def safe_load(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def infer_fields(p: Path, payload: dict):
    """Best-effort extraction that works for lexical / fused / sroberta runs."""
    emb_type = payload.get("embedding_type", "")
    model    = payload.get("model", "")
    seed     = int(payload.get("seed", 0))

    # lexical
    variant  = payload.get("variant", "")

    # sroberta / fused
    layer    = payload.get("sroberta_layer", payload.get("layer", ""))
    pool     = payload.get("pooling", payload.get("pool", ""))

    fused_mode = payload.get("fused_mode", "")
    lex_name   = payload.get("lex", payload.get("lex_name", ""))

    # num classes (optional)
    metrics = payload.get("metrics", {})
    return {
        "embedding_type": emb_type,
        "variant": variant,
        "layer": layer,
        "pool": pool,
        "model": model,
        "seed": seed,
        "fused_mode": fused_mode,
        "lex": lex_name,
        "accuracy": metrics.get("accuracy", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "weighted_f1": metrics.get("weighted_f1", np.nan),
        "macro_precision": metrics.get("macro_precision", np.nan),
        "macro_recall": metrics.get("macro_recall", np.nan),
        "path": str(p)
    }

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["embedding_type","variant","layer","pool","model","fused_mode","lex"]
    out = (df.groupby(group_cols, dropna=False)
             .agg(seeds=("seed","nunique"),
                  f1m_mean=("macro_f1","mean"), f1m_std=("macro_f1","std"),
                  f1w_mean=("weighted_f1","mean"), f1w_std=("weighted_f1","std"),
                  acc_mean=("accuracy","mean"), acc_std=("accuracy","std"))
             .reset_index())
    return out

def show_top(aggs: pd.DataFrame, k: int, key_mean: str, key_std: str, title: str):
    print(f"\n=== TOP-{k} by {title} ===")
    cols = ["embedding_type","variant","layer","pool","fused_mode","lex","model","seeds",
            key_mean, key_std, "acc_mean","acc_std"]
    print(aggs.sort_values(key_mean, ascending=False).head(k)[[c for c in cols if c in aggs.columns]].to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="result roots to scan")
    ap.add_argument("--save_rows_csv", default="", help="save raw rows here")
    ap.add_argument("--save_csv", default="", help="save aggregated summary here")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = []
    for r in args.roots:
        for p in Path(r).rglob("results.json"):
            payload = safe_load(p)
            if not payload: 
                continue
            rows.append(infer_fields(p, payload))

    if not rows:
        print("[WARN] no results.json found under given roots")
        return

    df = pd.DataFrame(rows).dropna(subset=["macro_f1","weighted_f1","accuracy"], how="all")
    if args.save_rows_csv:
        Path(args.save_rows_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_rows_csv, index=False)
        print(f"[OK] saved rows → {args.save_rows_csv}")

    aggs = aggregate(df)
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        aggs.to_csv(args.save_csv, index=False)
        print(f"[OK] saved summary → {args.save_csv}")

    show_top(aggs, args.top, "f1m_mean", "f1m_std", "macro_f1 (mean over seeds)")
    show_top(aggs, args.top, "f1w_mean", "f1w_std", "weighted_f1 (mean over seeds)")

if __name__ == "__main__":
    main()
