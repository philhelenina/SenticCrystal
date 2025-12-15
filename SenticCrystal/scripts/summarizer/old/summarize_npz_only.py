#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def load_json(p):
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None

def rows_from_run(run_dir: Path):
    meta = load_json(run_dir / "train_log.json") or {}
    out = []
    for which in ["mlp","lstm"]:
        mf = run_dir / f"{which}_metrics.json"
        if not mf.exists(): 
            continue
        m = load_json(mf) or {}
        # 키 이름 유연 처리
        acc = m.get("accuracy", m.get("acc"))
        f1m = m.get("macro_f1", m.get("macro-F1"))
        f1w = m.get("weighted_f1", m.get("weighted-F1"))
        out.append(dict(
            embedding_type = meta.get("embedding_type",""),
            layer          = meta.get("layer",""),
            pool           = meta.get("pool",""),
            variant        = meta.get("variant",""),
            model          = which,
            seed           = meta.get("seed", 0),
            accuracy       = acc,
            macro_f1       = f1m,
            weighted_f1    = f1w,
            path           = str(mf)
        ))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--save_csv", default="")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for r in args.roots:
        root = Path(r)
        # train_log.json을 가진 폴더만 대상
        for meta in root.rglob("train_log.json"):
            rows += rows_from_run(meta.parent)

    if not rows:
        print("[WARN] no NPZ runs found under:", args.roots)
        return

    df = pd.DataFrame(rows).dropna(subset=["macro_f1"])
    # 집계: 동일 설정(임베딩/레이어/풀/모델)별 seed 평균±표준편차
    grp = ["embedding_type","layer","pool","variant","model"]
    aggs = df.groupby(grp).agg(
        seeds=("seed","nunique"),
        f1m_mean=("macro_f1","mean"),
        f1m_std =("macro_f1","std"),
        f1w_mean=("weighted_f1","mean"),
        f1w_std =("weighted_f1","std"),
        acc_mean=("accuracy","mean"),
        acc_std =("accuracy","std"),
    ).reset_index()

    def show_top(aggs, by, title, k):
        if aggs.empty:
            print(f"\n=== TOP-{k} by {title} ===\n(no data)")
            return
        cols = ["embedding_type","layer","pool","variant","model","seeds",by,by.replace("mean","std"),"acc_mean","acc_std"]
        print(f"\n=== TOP-{k} by {title} ===")
        print(aggs.sort_values(by, ascending=False).head(k)[cols].to_string(index=False))

    show_top(aggs, "f1m_mean", "macro_f1 (mean over seeds)", args.top)
    show_top(aggs, "f1w_mean", "weighted_f1 (mean over seeds)", args.top)

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        aggs.to_csv(out, index=False)
        print(f"\n[OK] saved → {out}")

if __name__ == "__main__":
    main()
