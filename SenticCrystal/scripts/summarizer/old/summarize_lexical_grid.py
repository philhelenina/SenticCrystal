#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_lexical_grid.py
- results/lexical_grid/ 아래 조합별 out_dir에서 *_metrics.json 수집
- seed 평균±표준편차를 집계하여 CSV/LaTeX 테이블 생성
- (옵션) blend만 추려서 alpha-tau pivot도 저장
"""
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RES_BASE = ROOT / "results" / "lexical_grid"
OUT_DIR = RES_BASE / "_summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for dirpath, _, files in os.walk(RES_BASE):
    dirpath = Path(dirpath)
    if "_summaries" in str(dirpath): 
        continue
    for fn in files:
        if not fn.endswith("_metrics.json"):
            continue
        f = dirpath / fn
        try:
            data = json.load(open(f, "r"))
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
            continue

        # 경로 파싱: .../lexical_grid/{idf|noidf}/a{A}_t{T}_{idf|noidf}/{variant}_mlp/...
        parts = f.parts
        if "lexical_grid" not in parts:
            continue
        try:
            idx = parts.index("lexical_grid")
            idf_flag = parts[idx+1]         # idf | noidf
            tag = parts[idx+2]              # a{A}_t{T}_{idf|noidf}
            var_dir = parts[idx+3]          # {variant}_mlp
        except Exception:
            idf_flag, tag, var_dir = "", "", ""

        # tag에서 alpha, tau 추출
        alpha = tau = None
        try:
            # tag: a0.2_t0.1_idf
            t1 = tag.split("_")
            alpha = float(t1[0].replace("a",""))
            tau = float(t1[1].replace("t",""))
        except Exception:
            pass

        variant = var_dir.replace("_mlp","")
        seed = data.get("seed", None)

        metrics = data.get("metrics", {})
        rows.append(dict(
            idf=idf_flag, alpha=alpha, tau=tau, variant=variant, seed=seed,
            acc=metrics.get("accuracy"), macro_f1=metrics.get("macro_f1"),
            weighted_f1=metrics.get("weighted_f1"),
            macro_precision=metrics.get("macro_precision"),
            macro_recall=metrics.get("macro_recall"),
            path=str(f)
        ))

df = pd.DataFrame(rows)
if df.empty:
    print("[WARN] no metrics found under", RES_BASE)
    raise SystemExit()

# seed-avg 요약
agg = (df
       .groupby(["idf","variant","alpha","tau"])
       .agg(acc_mean=("acc","mean"), acc_std=("acc","std"),
            macro_f1_mean=("macro_f1","mean"), macro_f1_std=("macro_f1","std"),
            weighted_f1_mean=("weighted_f1","mean"), weighted_f1_std=("weighted_f1","std"),
            n=("macro_f1","count"))
       .reset_index()
      ).sort_values(["idf","variant","alpha","tau"])

# CSV/LaTeX 저장
csv_path = OUT_DIR / "lexical_grid_seedavg.csv"
agg.to_csv(csv_path, index=False)

def pm(m,s): 
    return f"{m:.4f} $\\pm$ {s:.4f}" if pd.notna(s) else f"{m:.4f}"

# 표: blend / blend+hist / hist-soft 각각
for subset_name, keep in {
    "blend": agg[agg["variant"]=="w2v-wna-blend"],
    "blend+hist": agg[agg["variant"]=="w2v-wna-blend+hist"],
    "hist-soft": agg[agg["variant"]=="wna-hist-soft"],
    "avg": agg[agg["variant"]=="w2v-avg"]
}.items():
    if keep.empty: 
        continue
    tex_lines = []
    tex_lines += [r"\begin{table}[t]", r"\centering", r"\small",
                  rf"\caption{{Lexical ({subset_name}) grid on IEMOCAP (4-way). Macro-F1 mean$\pm$std over seeds.}}",
                  rf"\label{{tab:lexical-{subset_name}-grid}}",
                  r"\begin{tabular}{lcccc}", r"\toprule",
                  r"IDF & $\alpha$ & $\tau$ & Macro-F1 & Acc \\",
                  r"\midrule"]
    for _, r in keep.iterrows():
        tex_lines.append(f"{r['idf']} & {r['alpha']:.2f} & {r['tau']:.2f} & "
                         f"{pm(r['macro_f1_mean'], r['macro_f1_std'])} & "
                         f"{pm(r['acc_mean'], r['acc_std'])} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / f"table_{subset_name}_grid.tex").write_text("\n".join(tex_lines))

# 히트맵용 pivot(variant=blend만)
blend = agg[(agg["variant"]=="w2v-wna-blend")].copy()
if not blend.empty:
    for idf_flag in ["idf","noidf"]:
        sub = blend[blend["idf"]==idf_flag]
        if sub.empty: 
            continue
        piv = sub.pivot(index="alpha", columns="tau", values="macro_f1_mean").sort_index()
        piv.to_csv(OUT_DIR / f"pivot_blend_macroF1_{idf_flag}.csv")

print("[OK] saved:")
print("-", csv_path)
for f in OUT_DIR.glob("table_*_grid.tex"):
    print("-", f)
for f in OUT_DIR.glob("pivot_blend_macroF1_*.csv"):
    print("-", f)
