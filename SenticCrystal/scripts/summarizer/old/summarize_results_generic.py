#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_results_generic.py
- 주어진 루트(들) 아래의 results.json들을 재귀 탐색해 수집/요약
- 6-way lexical grid, lexical baseline, sroberta 등 공통 포맷 지원
- 출력: rows CSV(개별 run) + summary CSV(시드평균) + 콘솔 Top-K

사용 예:
python3 summarize_results_generic.py \
  --roots /.../results/6way/npz_lexical_grid \
          /.../results/6way/npz_lexical_idf \
          /.../results/6way/npz_lexical_noidf \
  --save_rows_csv /.../results/6way/all_rows_6way.csv \
  --save_summary_csv /.../results/6way/all_summary_6way.csv \
  --top 20

ROOT="/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment"

# 1) 6-way lexical grid + 기존 lexical 결과를 한 번에
python3 "$ROOT/scripts/summarize_results_generic.py" \
  --roots \
    "$ROOT/results/6way/npz_lexical_grid" \
    "$ROOT/results/6way/npz_lexical_idf" \
    "$ROOT/results/6way/npz_lexical_noidf" \
  --save_rows_csv    "$ROOT/results/6way/all_rows_6way.csv" \
  --save_summary_csv "$ROOT/results/6way/all_summary_6way.csv" \
  --top 20

# 2) sroberta / hier / fused 등 다른 결과 폴더도 원하면 roots에 추가
# 예: "$ROOT/results/6way/npz_sr" "$ROOT/results/6way/npz_sr_hier" "$ROOT/results/6way/npz_fused"
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd
import numpy as np

def find_results(roots):
    files=[]
    for r in roots:
        r=Path(r)
        files += list(r.rglob("results.json"))
    return files

TAG_RE = re.compile(r"a(?P<alpha>[\d.]+)_t(?P<tau>[\d.]+)_b(?P<beta>[\d.]+)")

def parse_one(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None

    d = {}
    # 기본 메타(없으면 경로로 추론)
    d["embedding_type"] = obj.get("embedding_type", infer_by_path(path, ["sentence-roberta","lexical"]))
    d["model"]          = obj.get("model", infer_by_path(path, ["mlp","lstm","mean","attn"]))
    d["seed"]           = obj.get("seed", infer_seed(path))

    # lexical 계열 추가 메타
    d["lex_set"]  = obj.get("lex_set", infer_by_path(path, ["idf","noidf"]))
    d["variant"]  = obj.get("variant", infer_variant(path))

    # variant 쪼개기 (e.g., "w2v-wna-blend/a0.7_t0.20_b0.2")
    if d["variant"]:
        parts = str(d["variant"]).split("/")
        d["variant_base"] = parts[0]
        d["tag"] = parts[1] if len(parts) > 1 else ""
        m = TAG_RE.fullmatch(d["tag"]) if d["tag"] else None
        d["alpha"] = float(m.group("alpha")) if m else np.nan
        d["tau"]   = float(m.group("tau"))   if m else np.nan
        d["beta"]  = float(m.group("beta"))  if m else np.nan
    else:
        d["variant_base"] = ""
        d["tag"] = ""
        d["alpha"]=d["tau"]=d["beta"]=np.nan

    # sroberta 계열 추가 메타(있을 때만)
    d["layer"]     = obj.get("layer", infer_by_path(path, ["last","avg_last4","last4_scalar_up","last4_scalar_down","last4_scalar_top2"]))
    d["pool"]      = obj.get("pool",  infer_by_path(path, ["cls","mean","attn","wmean_pos","wmean_pos_rev","wmean_idf"]))
    d["fused_mode"]= obj.get("fused_mode","")

    # metrics
    met = obj.get("metrics", {})
    d["acc"]  = met.get("accuracy",  np.nan)
    d["f1m"]  = met.get("macro_f1",  np.nan)
    d["f1w"]  = met.get("weighted_f1", np.nan)
    d["macro_precision"] = met.get("macro_precision", np.nan)
    d["macro_recall"]    = met.get("macro_recall",    np.nan)

    # 경로 기록
    d["run_dir"] = str(path.parent.resolve())
    return d

def infer_by_path(path: Path, candidates: list[str]) -> str:
    s = str(path)
    for c in candidates:
        if f"/{c}/" in s or s.endswith(f"/{c}/results.json"):
            return c
    return ""

def infer_seed(path: Path) -> int | str:
    m = re.search(r"seed(\d+)", str(path))
    return int(m.group(1)) if m else ""

def infer_variant(path: Path) -> str:
    # .../lexical_{idf|noidf}/<variant>(/<tag>)/{mlp|lstm}/seedXX/results.json
    s = str(path)
    m = re.search(r"/lexical(?:_idf|_noidf|)/([^/]+(?:/a[\d.]+_t[\d.]+_b[\d.]+)?)\/(mlp|lstm)\/seed\d+\/results\.json$", s)
    if m:
        return m.group(1)
    # 구버전: .../lexical_{idf|noidf}/<variant>/{mlp|lstm}/seedXX/results.json
    m2 = re.search(r"/lexical(?:_idf|_noidf|)/([^/]+)/(?:mlp|lstm)/seed\d+/results\.json$", s)
    return m2.group(1) if m2 else ""

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # seed 평균/표준편차: lexical은 [lex_set, variant_base, tag, model] 기준
    keys = []
    if (df["embedding_type"]=="lexical").any():
        keys = ["embedding_type","lex_set","variant_base","tag","model"]
    else:
        # sroberta 계열 (layer/pool 포함)
        keys = ["embedding_type","layer","pool","model"]

    g = (df.groupby(keys, dropna=False)
           .agg(seeds=("seed", "nunique"),
                f1m_mean=("f1m","mean"), f1m_std=("f1m","std"),
                f1w_mean=("f1w","mean"), f1w_std=("f1w","std"),
                acc_mean=("acc","mean"), acc_std=("acc","std"),
                alpha=("alpha","first"), tau=("tau","first"), beta=("beta","first"))
           .reset_index())
    # NaN std → 0
    for c in ["f1m_std","f1w_std","acc_std"]:
        g[c] = g[c].fillna(0.0)
    return g

def print_top(g: pd.DataFrame, k: int=20, by: str="f1m_mean"):
    cols = [c for c in ["embedding_type","lex_set","variant_base","tag","layer","pool","model","seeds","f1m_mean","f1m_std","f1w_mean","f1w_std","acc_mean","acc_std"] if c in g.columns]
    print(f"\n=== TOP-{k} by {by} ===")
    print(g.sort_values(by=by, ascending=False).head(k)[cols].to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="results.json이 들어있는 상위 폴더(여러 개 가능)")
    ap.add_argument("--save_rows_csv", required=True)
    ap.add_argument("--save_summary_csv", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    files = find_results(args.roots)
    if not files:
        print("[WARN] no runs found under:", args.roots); return

    rows = []
    for f in files:
        rec = parse_one(Path(f))
        if rec is not None:
            rows.append(rec)

    if not rows:
        print("[WARN] could not parse any results.json"); return

    df = pd.DataFrame(rows)
    df.to_csv(args.save_rows_csv, index=False)
    print(f"[OK] rows → {args.save_rows_csv}  (n={len(df)})")

    g = aggregate(df)
    g.to_csv(args.save_summary_csv, index=False)
    print(f"[OK] summary → {args.save_summary_csv}  (n={len(g)})")

    # Top-K 출력
    if "f1m_mean" in g.columns:
        print_top(g, args.top, "f1m_mean")
    if "f1w_mean" in g.columns:
        print_top(g, args.top, "f1w_mean")

if __name__ == "__main__":
    main()
