#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_hier_lexical_results.py
- Collect results.json under results/hier_lexical
- Save raw + seed-averaged CSV
- Print top models by task (4way, 6way) to terminal
"""

import json
from pathlib import Path
import pandas as pd

HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
RES  = HOME/"results"/"hier_lexical"
OUT  = HOME/"results"/"_summaries"
OUT.mkdir(parents=True, exist_ok=True)

def safe_tag(s: str) -> str:
    return s.replace("/", "_").replace("+", "_").replace(":", "_")

rows=[]
for task_dir in (RES/"4way", RES/"6way"):
    if not task_dir.exists():
        continue
    task = task_dir.name
    for json_file in task_dir.rglob("results.json"):
        try:
            with open(json_file,"r") as f:
                j=json.load(f)
            m = j.get("metrics", {})
            parent = json_file.parent
            parts = parent.parts
            idx = parts.index("hier_lexical")
            rel = parts[idx+1:]
            # rel = [task, root_tag, mode, layer, pool, model, seedXX]
            task = rel[0]
            root_tag = "/".join(rel[1:-5])
            mode, layer, pool, model, seed_dir = rel[-5:]
            seed = int(seed_dir.replace("seed",""))

            rows.append(dict(
                task=task,
                root_tag=safe_tag(root_tag),
                mode=mode, layer=layer, pool=pool, model=model, seed=seed,
                acc=m.get("accuracy"),
                f1w=m.get("weighted_f1"),
                f1m=m.get("macro_f1")
            ))
        except Exception as e:
            print(f"[WARN] skip {json_file}: {e}")

df = pd.DataFrame(rows).sort_values(["task","root_tag","mode","layer","pool","model","seed"])
raw_file = OUT/"hier_lexical_raw.csv"
df.to_csv(raw_file, index=False)

if not df.empty:
    grp = df.groupby(["task","root_tag","mode","layer","pool","model"], as_index=False)\
            .agg(acc=("acc","mean"), f1w=("f1w","mean"), f1m=("f1m","mean"),
                 seeds=("seed","nunique"))
    avg_file = OUT/"hier_lexical_seedavg.csv"
    grp.to_csv(avg_file, index=False)
    print("[OK] saved:", raw_file, "|", avg_file)

    # === 터미널 출력: Task별 Top-K ===
    TOPK = 10
    for task in ["4way","6way"]:
        sub = grp[grp["task"]==task]
        if sub.empty: 
            continue
        print(f"\n=== {task.upper()} Top {TOPK} Models by Weighted F1 ===")
        print(sub.sort_values("f1w", ascending=False).head(TOPK).to_string(index=False))

        print(f"\n=== {task.upper()} Top {TOPK} Models by Macro F1 ===")
        print(sub.sort_values("f1m", ascending=False).head(TOPK).to_string(index=False))
else:
    print("[WARN] no results found under", RES)
