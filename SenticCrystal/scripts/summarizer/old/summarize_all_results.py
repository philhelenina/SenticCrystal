#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_all_results.py
- 여러 결과 루트에서 *_metrics.json 또는 *_baseline_results*.json 수집
- seed-avg(평균±표준편차) 집계 및 CSV/LaTeX 테이블 생성

사용:
  python scripts/summarize_all_results.py \
    --roots results/npz_baselines results/npz_baselines_idf results/npz_baselines_noidf \
            results/baseline_classifiers_ablation results/lexical_grid
"""

import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")

def pm(m, s, k=4):
    return f"{m:.{k}f} $\\pm$ {s:.{k}f}" if pd.notna(s) else f"{m:.{k}f}"

def sniff_seed_from_name(p: Path):
    # ..._seed42... or ..._seed=42...
    m = re.search(r"seed[_=]?(\d+)", p.stem)
    return int(m.group(1)) if m else None

def load_one_json(path: Path):
    try:
        data = json.load(open(path, "r"))
    except Exception as e:
        print(f"[WARN] skip {path}: {e}")
        return None

    # 공통 필드 최대한 추출
    # train_npz_classifier.py → *_metrics.json  (metrics:{...})
    # train_npy_classifier.py → *_baseline_results*.json (metrics:{...})
    metrics = data.get("metrics", data)
    emb_type = data.get("embedding_type")
    model    = data.get("model")
    pool     = data.get("pool") or data.get("pooling")
    layer    = data.get("layer", "")
    variant  = data.get("variant", "")
    seed     = data.get("seed", sniff_seed_from_name(path))

    # idf/noidf 태그는 경로에서 추정
    pstr = str(path)
    set_tag = ("idf" if "lexical_idf" in pstr or "/idf/" in pstr else
               "noidf" if "lexical_noidf" in pstr or "/noidf/" in pstr else "")

    # lexical_grid일 때 a{alpha}_t{tau} 추출
    alpha = tau = None
    m = re.search(r"/a([0-9.]+)_t([0-9.]+)_", pstr)
    if m:
        try:
            alpha = float(m.group(1)); tau = float(m.group(2))
        except: pass

    return dict(
        path=str(path),
        embedding_type=emb_type,
        model=(model.upper() if isinstance(model, str) else model),
        pool=pool,
        layer=layer,
        variant=variant,
        seed=seed,
        set_tag=set_tag,
        alpha=alpha, tau=tau,
        acc=metrics.get("accuracy"),
        macro_f1=metrics.get("macro_f1"),
        weighted_f1=metrics.get("weighted_f1"),
        macro_precision=metrics.get("macro_precision"),
        macro_recall=metrics.get("macro_recall"),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="결과 루트 디렉터리들 (재귀 검색)")
    ap.add_argument("--out_dir", default=str(HOME / "results" / "_summaries"),
                    help="요약 산출물 저장 폴더")
    args = ap.parse_args()

    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    print("[Scan] roots:", args.roots)
    for root in args.roots:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] skip missing root: {root}")
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.endswith("_metrics.json") or "baseline_results" in fn:
                    rec = load_one_json(Path(dirpath) / fn)
                    if rec:
                        rows.append(rec)

    raw = pd.DataFrame(rows)
    if raw.empty:
        print("[WARN] no results found.")
        return

    raw_path = OUT / "all_runs_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"[OK] saved raw → {raw_path}")

    # ===== seed-avg 집계 =====
    group_cols = ["embedding_type","variant","layer","pool","model","set_tag","alpha","tau"]
    agg = (raw
           .groupby(group_cols, dropna=False)
           .agg(acc_mean=("acc","mean"), acc_std=("acc","std"),
                macro_f1_mean=("macro_f1","mean"), macro_f1_std=("macro_f1","std"),
                weighted_f1_mean=("weighted_f1","mean"), weighted_f1_std=("weighted_f1","std"),
                n=("macro_f1","count"))
           .reset_index()
           .sort_values(group_cols)
          )
    seedavg_path = OUT / "all_runs_seedavg.csv"
    agg.to_csv(seedavg_path, index=False)
    print(f"[OK] saved seed-avg → {seedavg_path}")

    # ===== 자주 쓰는 LaTeX 표들 =====

    # (A) Sentence-RoBERTa avg_last4 (pool×model)
    sr = agg[(agg.embedding_type=="sentence-roberta") & (agg.layer.fillna("")=="avg_last4")].copy()
    if not sr.empty:
        sr["pool"] = pd.Categorical(sr["pool"], ["mean","cls","attn"], ordered=True)
        sr["model"]= pd.Categorical(sr["model"], ["MLP","LSTM"], ordered=True)
        sr = sr.sort_values(["model","pool"])
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{Sentence-RoBERTa (avg\_last4) baselines on IEMOCAP (4-way). Mean$\pm$std over seeds.}",
                 r"\label{tab:npy-sr-baselines}", r"\begin{tabular}{llccc}", r"\toprule",
                 r"Model & Pool & Macro-F1 & Acc & Weighted-F1 \\", r"\midrule"]
        for _, r in sr.iterrows():
            lines.append(f"{r['model']} & {r['pool']} & "
                         f"{pm(r['macro_f1_mean'], r['macro_f1_std'])} & "
                         f"{pm(r['acc_mean'], r['acc_std'])} & "
                         f"{pm(r['weighted_f1_mean'], r['weighted_f1_std'])} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (OUT / "table_sr_avg_last4.tex").write_text("\n".join(lines))
        print("[OK] table →", OUT / "table_sr_avg_last4.tex")

    # (B) NPY 베이스라인: 임베딩×풀링×모델
    npy = agg[(agg.embedding_type.isin(["bert","roberta","emo-roberta","sentence-roberta"]))
              & (agg.layer.isna()) & (agg.variant.isna() | (agg.variant==""))].copy()
    if not npy.empty:
        npy["pool"] = pd.Categorical(npy["pool"], ["mean","cls","attn"], ordered=True)
        npy["model"]= pd.Categorical(npy["model"], ["MLP","LSTM"], ordered=True)
        npy = npy.sort_values(["embedding_type","model","pool"])
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{NPY baselines (BERT/ RoBERTa/ Emo-RoBERTa/ Sentence-RoBERTa). Mean$\pm$std over seeds.}",
                 r"\label{tab:npy-all-baselines}", r"\begin{tabular}{lllccc}", r"\toprule",
                 r"Embed & Model & Pool & Macro-F1 & Acc & Weighted-F1 \\", r"\midrule"]
        for _, r in npy.iterrows():
            lines.append(f"{r['embedding_type']} & {r['model']} & {r['pool']} & "
                         f"{pm(r['macro_f1_mean'], r['macro_f1_std'])} & "
                         f"{pm(r['acc_mean'], r['acc_std'])} & "
                         f"{pm(r['weighted_f1_mean'], r['weighted_f1_std'])} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (OUT / "table_npy_all_baselines.tex").write_text("\n".join(lines))
        print("[OK] table →", OUT / "table_npy_all_baselines.tex")

    # (C) Lexical ablation (IDF/NOIDF 블록, MLP만)
    lex = agg[(agg.embedding_type=="lexical") & (agg.model=="MLP")].copy()
    if not lex.empty:
        for tag in ["idf","noidf",""]:
            sub = lex[lex["set_tag"]==tag] if tag else lex[lex["set_tag"].isin(["","idf","noidf"])]
            if sub.empty: continue
            sub = sub.sort_values(["variant","alpha","tau"])
            cap_tag = (tag.upper() if tag else "all")
            lines = [r"\begin{table}[t]", r"\centering", r"\small",
                     rf"\caption{{Lexical ablations ({cap_tag}). Mean$\pm$std over seeds.}}",
                     rf"\label{{tab:lexical-ablations-{cap_tag.lower()}}}",
                     r"\begin{tabular}{lcccc}", r"\toprule",
                     r"Variant & Macro-F1 & Acc & Weighted-F1 & n \\", r"\midrule"]
            for _, r in sub.iterrows():
                lines.append(f"{r['variant']} & "
                             f"{pm(r['macro_f1_mean'], r['macro_f1_std'])} & "
                             f"{pm(r['acc_mean'], r['acc_std'])} & "
                             f"{pm(r['weighted_f1_mean'], r['weighted_f1_std'])} & "
                             f"{int(r['n'])} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            (OUT / f"table_lexical_ablations_{cap_tag.lower()}.tex").write_text("\n".join(lines))
            print("[OK] table →", OUT / f"table_lexical_ablations_{cap_tag.lower()}.tex")

    print("[DONE] summaries saved under:", OUT)

if __name__ == "__main__":
    main()
