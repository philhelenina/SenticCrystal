#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_baselines.py
- Collect results.json from baseline runs (BERT, RoBERTa, EmoRoBERTa)
- Works for both 4way and 6way
- Outputs: CSV with raw rows + CSV with seed-averaged summary

python scripts/summarize_baselines.py \
  --root results/baselines \
  --out_dir results/_summaries
"""

import argparse, json
from pathlib import Path
import pandas as pd

def collect_results(root: Path):
    rows = []
    for f in root.rglob("results.json"):
        parts = f.parts
        # Expected structure: results/baselines/{4way,6way}/{model}/{pool}/seedXX/{mlp,lstm}/results.json
        try:
            task = parts[-6]  # 4way or 6way
            model = parts[-5]
            pool = parts[-4]
            seed = parts[-3]
            clf  = parts[-2]
        except IndexError:
            continue

        data = json.load(open(f))
        row = dict(
            task=task, model=model, pool=pool, seed=seed, clf=clf,
            acc=data["metrics"]["accuracy"],
            f1m=data["metrics"]["macro_f1"],
            f1w=data["metrics"]["weighted_f1"],
        )
        rows.append(row)
    return pd.DataFrame(rows)

def summarize(df):
    grouped = df.groupby(["task","model","pool","clf"])
    summary = grouped.agg(
        acc_mean=("acc","mean"), acc_std=("acc","std"),
        f1m_mean=("f1m","mean"), f1m_std=("f1m","std"),
        f1w_mean=("f1w","mean"), f1w_std=("f1w","std"),
        n=("acc","count")
    ).reset_index()
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/baselines",
                    help="Path to baseline results")
    ap.add_argument("--out_dir", type=str, default="results/_summaries",
                    help="Where to save CSV outputs")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(root)
    raw_csv = out_dir / "baselines_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"[OK] Raw rows → {raw_csv}")

    summary = summarize(df)
    sum_csv = out_dir / "baselines_seedavg.csv"
    summary.to_csv(sum_csv, index=False)
    print(f"[OK] Summary (seed-avg) → {sum_csv}")

if __name__ == "__main__":
    main()
