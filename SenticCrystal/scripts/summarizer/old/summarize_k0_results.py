#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_k0_results.py  (robust, fused + NPZ)
- fused(NPY): .../fused_*/<lex>/<layer_pool>/<model>/seedXX/results.json
- NPZ:       .../<run_dir>/{train_log.json, mlp_metrics.json, lstm_metrics.json}
- 평균±표준편차 집계, Top-K 출력, CSV 저장

예시:
  python3 scripts/summarize_k0_results.py \
    --roots \
      /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/npz_baselines_sr_newpools \
      /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/npz_baselines_sr_newpools_layers \      
      /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/w2v-idf-sentence-roberta-fused \
      /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/w2v-sentence-roberta-fused \
    --scan auto \
    --save_csv /home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/k0_all_summary.csv \
    --top 10 --print_counts

ROOT="/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment"

전체 뽑으려면:
python3 "$ROOT/scripts/summarize_k0_results.py" \
  --scan npz \
  --roots \
    "$ROOT/results/npz_baselines_sr_newpools" \
    "$ROOT/results/npz_baselines_sr_newpools_last" \
    "$ROOT/results/npz_baselines_sr_newpools_layers" \
  --save_csv "$ROOT/results/sr_npz_summary.csv" \
  --save_rows "$ROOT/results/sr_npz_rows.csv" \
  --top 20


"""

from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd

# ---------------- Utils ----------------
def safe_load_json(p: Path):
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None

def gkeys(d: dict, *keys, default=None):
    for k in keys:
        if k in d: return d[k]
    return default

# ---------------- Param counters ----------------
def count_params_mlp(in_dim: int, hidden: int, num_classes: int) -> int:
    return in_dim*hidden + hidden + hidden*num_classes + num_classes

def count_params_lstm(in_dim: int, hidden: int, num_classes: int) -> int:
    # PyTorch 1-layer uni LSTM: 4H(I+H) + 8H ; head: H*C + C
    lstm_core = 4*hidden*(in_dim + hidden) + 8*hidden
    head = hidden*num_classes + num_classes
    return lstm_core + head

# ---------------- Inference helpers ----------------
def infer_layer_pool_from_leaf(leaf: str):
    # examples: "avg_last4_cls", "avg_last4_wmean_pos", ...
    m = re.match(r"((?:last|avg_last4|last4_scalar(?:_(?:up|down|top2))?))_(cls|mean|attn|wmean_pos(?:_rev)?|wmean_idf)", leaf)
    if m:
        return m.group(1), m.group(2)
    layer = "avg_last4"
    if "last4" in leaf: layer="last4_scalar"
    elif "last" in leaf: layer="last"
    pool = "mean"
    for key in ["wmean_pos_rev","wmean_pos","wmean_idf","attn","cls","mean"]:
        if key in leaf:
            pool = key; break
    return layer, pool

def infer_in_dim_from_fused(parts: tuple) -> int:
    # .../fused_concat|fused_zeropad768|fused_proj128/<lex>/<layer_pool>/<model>/seedXX/results.json
    if len(parts) < 6:
        return 768 + 300
    fused_mode = parts[-6]
    lex = parts[-5]
    s_dim = 768
    l_dim = 300 if lex in ("w2v_avg","w2v_wna_blend") else 300
    if "zeropad768" in fused_mode:
        return s_dim + 768
    if "proj128" in fused_mode:
        return s_dim + 128
    return s_dim + l_dim

def try_infer_in_dim_npz(run_dir: Path, meta: dict) -> int:
    et = meta.get("embedding_type","")
    if et == "sentence-roberta":
        return 768
    if et == "lexical":
        # variant별 실제 파일 shape 확인
        variant = meta.get("variant","")
        # project 구조에 맞춰 상위로 거슬러 올라가며 'data/embeddings/4way' 찾기
        root = run_dir
        emb_base = None
        for _ in range(6):
            cand = root.parent / "data" / "embeddings" / "4way"
            if cand.exists():
                emb_base = cand; break
            root = root.parent
        if emb_base is not None:
            p = emb_base / "lexical" / variant / "train.npz"
            if p.exists():
                arr = np.load(p)
                X = arr["embeddings"] if "embeddings" in arr else arr[list(arr.keys())[0]]
                return int(X.shape[1])
        return 300
    return 768

# ---------------- Collectors ----------------
def collect_runs_fused(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for p in root.rglob("results.json"):
            payload = safe_load_json(p)
            if not payload:
                continue
            parts = p.parts
            # weak guard: ensure path includes fused_*
            if not any("fused_" in seg for seg in parts):
                continue

            # parse path
            try:
                model = parts[-3]  # mlp|lstm
                seedm = re.findall(r"seed(\d+)", parts[-2])
                seed  = int(seedm[0]) if seedm else int(gkeys(payload, "seed", default=0))
                layer, pool = infer_layer_pool_from_leaf(parts[-4])
                lex   = parts[-5]
                fused_mode = parts[-6].replace("fused_","")
            except Exception:
                continue

            metrics = payload.get("metrics", {})
            in_dim = infer_in_dim_from_fused(parts)
            C = 4
            hidden = int(gkeys(payload, "hidden_size", default=(256 if model=="mlp" else 128)))
            params = count_params_mlp(in_dim, hidden, C) if model=="mlp" else count_params_lstm(in_dim, hidden, C)

            rows.append(dict(
                source="fused",
                fused_mode=fused_mode,
                lex=lex, layer=layer, pool=pool, model=model,
                seed=seed, in_dim=in_dim, params=params,
                accuracy=gkeys(metrics, "accuracy", "acc"),
                macro_f1=gkeys(metrics, "macro_f1", "macro-F1"),
                weighted_f1=gkeys(metrics, "weighted_f1", "weighted-F1"),
                path=str(p)
            ))
    return pd.DataFrame(rows)

def collect_one_model_npz(metric_file: Path, run_dir: Path):
    payload = safe_load_json(metric_file)
    meta = safe_load_json(run_dir / "train_log.json") or {}
    if not payload:
        return []

    model  = "mlp" if metric_file.name.startswith("mlp_") else "lstm"
    seed   = int(meta.get("seed", 0))
    hidden = int(meta.get("hidden_size", 256 if model=="mlp" else 128))
    C      = len(meta.get("class_names", [])) or 4
    in_dim = try_infer_in_dim_npz(run_dir, meta)
    params = count_params_mlp(in_dim, hidden, C) if model=="mlp" else count_params_lstm(in_dim, hidden, C)

    return [dict(
        source="npz",
        fused_mode="",
        lex=meta.get("variant",""),
        layer=meta.get("layer",""),
        pool=meta.get("pool",""),
        model=model,
        seed=seed, in_dim=in_dim, params=params,
        accuracy=gkeys(payload, "accuracy", "acc"),
        macro_f1=gkeys(payload, "macro_f1", "macro-F1"),
        weighted_f1=gkeys(payload, "weighted_f1", "weighted-F1"),
        path=str(metric_file)
    )]

def collect_runs_npz(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        # 1) train_log.json 있는 디렉토리에서 수집
        for meta in root.rglob("train_log.json"):
            run_dir = meta.parent
            for mf in (run_dir / "mlp_metrics.json", run_dir / "lstm_metrics.json"):
                if mf.exists():
                    rows.extend(collect_one_model_npz(mf, run_dir))
        # 2) 혹시 train_log 없이 *_metrics.json만 있는 폴더도 수집
        for mf in root.rglob("*_metrics.json"):
            run_dir = mf.parent
            if (run_dir / "train_log.json").exists():
                continue
            rows.extend(collect_one_model_npz(mf, run_dir))
    return pd.DataFrame(rows)

# ---------------- Aggregation & Display ----------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["source","fused_mode","lex","layer","pool","model","in_dim","params"]
    aggs = df.groupby(grp_cols, dropna=False).agg(
        seeds=("seed","nunique"),
        f1m_mean=("macro_f1","mean"),
        f1m_std =("macro_f1","std"),
        f1w_mean=("weighted_f1","mean"),
        f1w_std =("weighted_f1","std"),
        acc_mean=("accuracy","mean"),
        acc_std =("accuracy","std")
    ).reset_index()
    aggs["params_M"] = (aggs["params"] / 1e6).round(3)
    return aggs

def show_top(aggs: pd.DataFrame, k: int, title: str, key_mean: str, key_std: str):
    print(f"\n=== TOP-{k} by {title} ===")
    if aggs.empty:
        print("(no data)")
        return
    cols = [c for c in ["source","fused_mode","lex","layer","pool","model","seeds","params_M","in_dim",key_mean,key_std,"acc_mean","acc_std"] if c in aggs.columns]
    print(aggs.sort_values(key_mean, ascending=False).head(k)[cols].to_string(index=False))

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="결과 루트(여러 개 가능)")
    ap.add_argument("--scan", choices=["auto","fused","npz"], default="auto",
                    help="스캔 대상 선택 (auto=둘 다, fused=NPY only, npz=NPZ only)")
    ap.add_argument("--save_csv", type=str, default="", help="집계 CSV 저장 경로")
    ap.add_argument("--save_rows_csv", type=str, default="", help="raw 행 CSV 저장(디버그용)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--print_counts", action="store_true")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]

    if args.scan in ("auto","fused"):
        df_fused = collect_runs_fused(roots)
    else:
        df_fused = pd.DataFrame()

    if args.scan in ("auto","npz"):
        df_npz = collect_runs_npz(roots)
    else:
        df_npz = pd.DataFrame()

    df = pd.concat([df_fused, df_npz], ignore_index=True)

    if args.print_counts:
        # 간단 카운트 출력
        print(f"[INFO] fused rows={len(df_fused)}  npz rows={len(df_npz)}  total={len(df)}")

    # NaN 메트릭 드롭
    if len(df)==0:
        print("[WARN] no runs found")
        return
    df = df.dropna(subset=["macro_f1", "weighted_f1", "accuracy"], how="all")

    if args.save_rows_csv:
        out = Path(args.save_rows_csv); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[OK] saved raw rows → {out}")

    if len(df)==0:
        print("[WARN] no valid metric rows")
        return

    aggs = aggregate(df)
    show_top(aggs, args.top, "macro_f1 (mean over seeds)", "f1m_mean", "f1m_std")
    show_top(aggs, args.top, "weighted_f1 (mean over seeds)", "f1w_mean", "f1w_std")

    if args.save_csv:
        out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
        aggs.to_csv(out, index=False)
        print(f"\n[OK] saved summary → {out}")

if __name__ == "__main__":
    main()

# python3 scripts/summarize_k0_results.py \
#  --roots #/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/npz_baselines_sr_newpools \
#  --save_csv #/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment/results/sr_newpools_summary.csv \
#  --top 10

