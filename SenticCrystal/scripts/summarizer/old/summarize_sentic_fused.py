#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_sentic_fused.py
- Collect results.json from SenticNet×SRoBERTa fusion runs and summarize.
- Supports both 4way and 6way layouts produced by sentic_train_all.sh:
  results/{TAG}/{TASK}/{MODE}/senticnet_axes/{LAYER}_{POOL}/{MODEL}/seed{SEED}/results.json

Outputs:
  - <out_dir>/sentic_fused_raw.csv          (each seed run)
  - <out_dir>/sentic_fused_seedavg.csv      (mean/std over seeds per group)
  - <out_dir>/top_by_weighted_f1_4way.csv   (best per LAYER/POOL/MODE/Model)
  - <out_dir>/top_by_weighted_f1_6way.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import re
import pandas as pd

HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RESULTS = HOME / "results"

# Pools we know (used to parse "layer_pool" safely)
POOLS = [
    "wmean_pos_rev", "wmean_pos", "wmean_idf",
    "attn", "mean", "cls"
]

def parse_layer_pool(layerpool: str) -> Tuple[str, str]:
    """Split '{layer}_{pool}' into (layer, pool) by matching known pool suffix."""
    for p in sorted(POOLS, key=lambda x: -len(x)):
        suf = f"_{p}"
        if layerpool.endswith(suf):
            return layerpool[:-len(suf)], p
    # fallback: couldn't split
    return layerpool, "unknown"

def read_metrics(json_path: Path) -> Optional[Dict[str, float]]:
    """Read results.json robustly (support both new and legacy shapes)."""
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # New schema: {"metrics": {...}, "args": {...}}
    if isinstance(obj, dict) and "metrics" in obj and isinstance(obj["metrics"], dict):
        m = obj["metrics"]
        # normalize keys
        acc = m.get("accuracy", m.get("acc"))
        f1m = m.get("macro_f1", m.get("f1m"))
        f1w = m.get("weighted_f1", m.get("f1w"))
        return {"accuracy": float(acc), "macro_f1": float(f1m), "weighted_f1": float(f1w)}
    # Legacy flat keys
    acc = obj.get("accuracy", obj.get("acc"))
    f1m = obj.get("macro_f1", obj.get("f1m"))
    f1w = obj.get("weighted_f1", obj.get("f1w"))
    if acc is None or f1m is None or f1w is None:
        return None
    return {"accuracy": float(acc), "macro_f1": float(f1m), "weighted_f1": float(f1w)}

def read_args(json_path: Path) -> Dict[str, Any]:
    """Grab args (if present) for aux columns like lex_scale."""
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict) and "args" in obj and isinstance(obj["args"], dict):
        return obj["args"]
    return {}

def collect_rows(results_root: Path, tags: List[str]) -> pd.DataFrame:
    """
    Scan results/{TAG}/{TASK}/{MODE}/senticnet_axes/{LAYER}_{POOL}/{MODEL}/seed{SEED}/results.json
    """
    rows = []
    for tag in tags:
        base = results_root / tag
        if not base.exists():
            # Some setups put task directly under tag (already in sentic_train_all.sh we did that)
            # We'll still try to iterate tasks later by globbing.
            pass
        # accept both explicit tasks and globbing
        for task_dir in base.glob("*"):
            task = task_dir.name  # "4way" or "6way"
            if task not in ("4way", "6way"):
                continue
            for mode_dir in task_dir.glob("*"):
                mode = mode_dir.name  # concat / proj128 / zeropad768
                sentic_root = mode_dir / "senticnet_axes"
                if not sentic_root.exists():
                    continue
                for lp_dir in sentic_root.glob("*"):
                    layerpool = lp_dir.name  # "{layer}_{pool}"
                    layer, pool = parse_layer_pool(layerpool)
                    for model_dir in lp_dir.glob("*"):  # mlp / lstm
                        model = model_dir.name
                        for seed_dir in model_dir.glob("seed*"):
                            m = re.match(r"seed(\d+)", seed_dir.name)
                            if not m: 
                                continue
                            seed = int(m.group(1))
                            js = seed_dir / "results.json"
                            if not js.exists():
                                continue
                            metrics = read_metrics(js)
                            if not metrics:
                                continue
                            args = read_args(js)
                            rows.append({
                                "tag": tag,
                                "task": task,
                                "mode": mode,
                                "layer": layer,
                                "pool": pool,
                                "model": model,
                                "seed": seed,
                                "accuracy": metrics["accuracy"],
                                "macro_f1": metrics["macro_f1"],
                                "weighted_f1": metrics["weighted_f1"],
                                # optional knobs if present
                                "lex_scale": args.get("lex_scale", None),
                                "loss": args.get("loss", None),
                                "focal_gamma": args.get("focal_gamma", None),
                                "focal_alpha": args.get("focal_alpha", None),
                            })
    return pd.DataFrame(rows)

def seed_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["tag","task","mode","layer","pool","model","lex_scale","loss","focal_gamma","focal_alpha"]
    out = (df
        .groupby(group_cols, dropna=False)
        .agg(accuracy_mean=("accuracy","mean"),
             accuracy_std=("accuracy","std"),
             macro_f1_mean=("macro_f1","mean"),
             macro_f1_std=("macro_f1","std"),
             weighted_f1_mean=("weighted_f1","mean"),
             weighted_f1_std=("weighted_f1","std"),
             seeds=("seed","nunique"))
        .reset_index()
        .sort_values(["task","weighted_f1_mean"], ascending=[True, False]))
    return out

def top_by_task(seedavg: pd.DataFrame, task: str, n: int = 30) -> pd.DataFrame:
    if seedavg.empty:
        return seedavg
    sub = seedavg[seedavg["task"]==task].copy()
    if sub.empty:
        return sub
    # one best per (layer,pool,mode,model,lex_scale,loss) by weighted_f1_mean
    key = ["layer","pool","mode","model","lex_scale","loss","focal_gamma","focal_alpha"]
    idx = sub.sort_values("weighted_f1_mean", ascending=False).groupby(key, dropna=False).head(1)
    return idx.sort_values("weighted_f1_mean", ascending=False).head(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default=str(RESULTS),
                    help="root folder that contains 'results/<TAG>/*'")
    ap.add_argument("--tags", nargs="*", default=[
        "senticnet-sroberta-fused-4way",
        "senticnet-sroberta-fused-6way",
    ], help="result tags under results/, e.g., senticnet-sroberta-fused-4way ...")
    ap.add_argument("--out_dir", type=str, default=str(RESULTS/"_summaries"/"sentic_fused"),
                    help="where to write CSVs")
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = collect_rows(results_root, args.tags)
    if df_raw.empty:
        print("[WARN] No results found. Check --results_root and --tags.")
        return

    # Save raw
    df_raw = df_raw.sort_values(["task","mode","layer","pool","model","seed"])
    raw_path = out_dir / "sentic_fused_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"[OK] raw → {raw_path}  rows={len(df_raw)}")

    # Seed averages
    df_avg = seed_aggregate(df_raw)
    avg_path = out_dir / "sentic_fused_seedavg.csv"
    df_avg.to_csv(avg_path, index=False)
    print(f"[OK] seedavg → {avg_path}  rows={len(df_avg)}")

    # Top lists per task
    top4 = top_by_task(df_avg, "4way", n=args.topn)
    top6 = top_by_task(df_avg, "6way", n=args.topn)
    top4_path = out_dir / "top_by_weighted_f1_4way.csv"
    top6_path = out_dir / "top_by_weighted_f1_6way.csv"
    top4.to_csv(top4_path, index=False)
    top6.to_csv(top6_path, index=False)
    print(f"[OK] top tables → {top4_path.name}, {top6_path.name}")

    # Quick console heads
    print("\n=== [HEAD] 4-way top ===")
    print(top4.head(10).to_string(index=False))
    print("\n=== [HEAD] 6-way top ===")
    print(top6.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
