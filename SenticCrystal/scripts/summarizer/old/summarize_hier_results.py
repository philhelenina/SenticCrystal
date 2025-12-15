#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_hier_results.py
- 대상: results/sr_hier/<layer>/<pool>/<model>/seedXX/results.json
- 기능: seed별 runs 수집 → 평균/표준편차 집계 → Top-K 출력 → CSV 저장
- 모델 파라미터 수 추정 포함(비신경식 집계는 Head MLP만 계산)
"""

from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd

# ----- 안전 로딩 -----
def safe_load_json(p: Path):
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None

def gkeys(d: dict, *keys, default=None):
    for k in keys:
        if k in d: return d[k]
    return default

# ----- 파라미터 카운트 -----
def count_params_mlp(in_dim: int, hidden: int, num_classes: int) -> int:
    # Linear(in_dim->hidden) + b + Linear(hidden->C) + b
    return in_dim*hidden + hidden + hidden*num_classes + num_classes

def count_params_attn_scorer(d: int) -> int:
    # scorer: Linear(d,d//2) + bias + Linear(d//2,1) + bias
    h = d // 2
    return d*h + h + h*1 + 1

def count_params_lstm(in_dim: int, hidden: int, num_classes: int) -> int:
    # 1-layer uni LSTM: 4H(I+H) + 8H (biases) ; head: H*C + C
    lstm_core = 4*hidden*(in_dim + hidden) + 8*hidden
    head = hidden*num_classes + num_classes
    return lstm_core + head

# ----- in_dim / num_classes 추정 -----
def infer_d_model_from_results(payload: dict, default_d=768) -> int:
    # sentence-roberta-base 계열이면 보통 768
    return int(gkeys(payload, "d_model", default=default_d))

def infer_num_classes(train_log: dict, payload: dict, default_c=4) -> int:
    cn = gkeys(train_log, "class_names", default=None)
    if isinstance(cn, list) and len(cn) > 0:
        return int(len(cn))
    return int(gkeys(payload, "num_classes", default=default_c))

# ----- 수집 -----
def collect_hier_rows(roots: list[Path], default_d=768) -> pd.DataFrame:
    rows = []
    for root in roots:
        for p in root.rglob("results.json"):
            payload = safe_load_json(p)
            if not payload: 
                continue
            parts = p.parts
            # .../<layer>/<pool>/<model>/seedXX/results.json
            try:
                model = parts[-3]
                seedm = re.findall(r"seed(\d+)", parts[-2]); seed = int(seedm[0]) if seedm else int(gkeys(payload,"seed",default=0))
                pool  = parts[-4]
                layer = parts[-5]
            except Exception:
                continue

            # sibling train_log.json에서 보조 정보 로딩
            train_log = safe_load_json(p.parent / "train_log.json") or {}

            # 메트릭
            metrics = payload.get("metrics", {})
            acc  = gkeys(metrics, "accuracy", "acc")
            f1m  = gkeys(metrics, "macro_f1", "macro-F1")
            f1w  = gkeys(metrics, "weighted_f1", "weighted-F1")
            if acc is None and f1m is None and f1w is None:
                continue

            # 하이퍼/노브
            hidden = int(gkeys(payload, "hidden_size", default=128))
            dropout = float(gkeys(payload, "dropout", default=gkeys(payload,"dropout_rate",default=0.0)))
            knobs = gkeys(payload, "agg_knobs", default={})
            lsep_tau = float(gkeys(knobs, "lsep_tau", default=np.nan))
            pmean_p  = float(gkeys(knobs, "pmean_p",  default=np.nan))
            decay_lambda  = float(gkeys(knobs, "decay_lambda", default=np.nan))
            decay_reverse = gkeys(knobs, "decay_reverse", default=None)

            # 차원/클래스/파라미터
            d_model = infer_d_model_from_results(payload, default_d=default_d)
            C = infer_num_classes(train_log, payload, default_c=4)

            # 파라미터 수
            if model in ("mean","sum","max","lsep","pmean","expdecay"):
                params = count_params_mlp(d_model, hidden, C)  # aggregator는 비학습
            elif model == "attn":
                params = count_params_attn_scorer(d_model) + count_params_mlp(d_model, hidden, C)
            elif model == "lstm":
                params = count_params_lstm(d_model, hidden, C)
            else:
                params = np.nan

            rows.append(dict(
                source="hier",
                layer=layer, pool=pool, model=model, seed=seed,
                hidden=hidden, dropout=dropout, d_model=d_model, num_classes=C,
                lsep_tau=lsep_tau, pmean_p=pmean_p, decay_lambda=decay_lambda, decay_reverse=decay_reverse,
                accuracy=acc, macro_f1=f1m, weighted_f1=f1w,
                params=params, params_M=(params/1e6 if pd.notna(params) else np.nan),
                path=str(p)
            ))
    return pd.DataFrame(rows)

# ----- 집계/출력 -----
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp = ["source","layer","pool","model","hidden","dropout","d_model","num_classes",
           "lsep_tau","pmean_p","decay_lambda","decay_reverse","params","params_M"]
    ag = df.groupby(grp, dropna=False).agg(
        seeds=("seed","nunique"),
        f1m_mean=("macro_f1","mean"),
        f1m_std =("macro_f1","std"),
        f1w_mean=("weighted_f1","mean"),
        f1w_std =("weighted_f1","std"),
        acc_mean=("accuracy","mean"),
        acc_std =("accuracy","std")
    ).reset_index()
    return ag

def show_top(aggs: pd.DataFrame, k: int, key_mean: str, key_std: str, title: str):
    print(f"\n=== TOP-{k} by {title} ===")
    if aggs.empty:
        print("(no data)"); return
    cols = ["source","layer","pool","model","hidden","seeds","params_M","d_model",
            key_mean, key_std, "acc_mean","acc_std","lsep_tau","pmean_p","decay_lambda","decay_reverse"]
    cols = [c for c in cols if c in aggs.columns]
    print(aggs.sort_values(key_mean, ascending=False).head(k)[cols].to_string(index=False))

# ----- 메인 -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="sr_hier 결과 루트(여러 개 가능)")
    ap.add_argument("--save_csv", type=str, default="", help="집계 CSV 저장 경로")
    ap.add_argument("--save_rows_csv", type=str, default="", help="raw 행 CSV 저장 경로(선택)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=768, help="임베딩 차원(기본 768)")
    ap.add_argument("--print_counts", action="store_true")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    df = collect_hier_rows(roots, default_d=args.d_model)

    if args.print_counts:
        print(f"[INFO] total rows = {len(df)}")

    if df.empty:
        print("[WARN] no runs found"); return

    # 메트릭 누락 제거
    df = df.dropna(subset=["macro_f1","weighted_f1","accuracy"], how="all")
    if args.save_rows_csv:
        out = Path(args.save_rows_csv); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False); print(f"[OK] saved raw rows → {out}")

    if df.empty:
        print("[WARN] no valid metric rows"); return

    aggs = aggregate(df)
    show_top(aggs, args.top, "f1m_mean", "f1m_std", "macro_f1 (mean over seeds)")
    show_top(aggs, args.top, "f1w_mean", "f1w_std", "weighted_f1 (mean over seeds)")

    if args.save_csv:
        out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
        aggs.to_csv(out, index=False); print(f"\n[OK] saved summary → {out}")

if __name__ == "__main__":
    main()
