#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_6way_sr_results.py
- 6-way SR 결과 폴더를 스캔해 seeds 집계(mean±std), Top-K 출력, CSV 저장
- 기대 경로: results_6way/sr_grid/<layer>_<pool>/<model>/seedXX/results.json

PY=python3
ROOT="/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment"
$PY "$ROOT/scripts/summarize_6way_sr_results.py" \
  --root "$ROOT/results_6way/sr_grid" \
  --save_csv "$ROOT/results_6way/sr_grid_summary.csv" \
  --save_rows "$ROOT/results_6way/sr_grid_rows.csv" \
  --top 20

"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def safe_load(p: Path):
    try:
        with open(p,"r") as f:
            return json.load(f)
    except Exception:
        return None

def collect(root: Path) -> pd.DataFrame:
    rows=[]
    for p in root.rglob("results.json"):
        parts = p.parts
        try:
            model = parts[-3]                 # mlp | lstm
            layer_pool = parts[-4]            # e.g., avg_last4_wmean_pos_rev
            seed_str = parts[-2]              # seed42
            seed = int(seed_str.replace("seed",""))
            layer, pool = layer_pool.split("_", 1)
        except Exception:
            continue

        payload = safe_load(p)
        if not payload: continue
        m = payload.get("metrics", {})
        rows.append(dict(
            layer=layer, pool=pool, model=model, seed=seed,
            accuracy=m.get("accuracy"), macro_f1=m.get("macro_f1"),
            weighted_f1=m.get("weighted_f1"), path=str(p),
        ))
    return pd.DataFrame(rows)

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["layer","pool","model"], dropna=False)
    ag = grp.agg(
        seeds=("seed","nunique"),
        f1m_mean=("macro_f1","mean"),
        f1m_std=("macro_f1","std"),
        f1w_mean=("weighted_f1","mean"),
        f1w_std=("weighted_f1","std"),
        acc_mean=("accuracy","mean"),
        acc_std=("accuracy","std"),
    ).reset_index()
    return ag

def show_top(ag: pd.DataFrame, k: int, key_mean: str, key_std: str, title: str):
    print(f"\n=== TOP-{k} by {title} ===")
    if ag.empty:
        print("(no data)"); return
    cols = ["layer","pool","model","seeds",key_mean,key_std,"acc_mean","acc_std"]
    print(ag.sort_values(key_mean, ascending=False).head(k)[cols].to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results_6way/sr_grid 루트")
    ap.add_argument("--save_csv", default="", help="집계 CSV 경로")
    ap.add_argument("--save_rows", default="", help="raw rows CSV 경로")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    df = collect(Path(args.root))
    if df.empty:
        print("[WARN] no results.json found")
        return

    if args.save_rows:
        Path(args.save_rows).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_rows, index=False)
        print(f"[OK] saved rows → {args.save_rows}")

    ag = aggregate(df)
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        ag.to_csv(args.save_csv, index=False)
        print(f"[OK] saved summary → {args.save_csv}")

    show_top(ag, args.top, "f1m_mean", "f1m_std", "macro_f1 (mean over seeds)")
    show_top(ag, args.top, "f1w_mean", "f1w_std", "weighted_f1 (mean over seeds)")

if __name__ == "__main__":
    main()
