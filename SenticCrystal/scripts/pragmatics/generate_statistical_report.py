#!/usr/bin/env python3
"""
Comprehensive Statistical Report Generator
Integrates dialogue stats, discourse markers, experimental design
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Define paths
HOME = Path("/home/jovyan/workspace/SenticCrystal/saturn_cloud_deployment")
STATS_DIR = HOME / "results" / "statistics"
MARKERS_DIR = HOME / "results" / "discourse_markers"

def load_analysis_results(stats_dir=None, markers_dir=None):
    """Load all analysis results"""
    if stats_dir is None:
        stats_dir = STATS_DIR
    if markers_dir is None:
        markers_dir = MARKERS_DIR
    
    stats_dir = Path(stats_dir)
    markers_dir = Path(markers_dir)
    
    results = {}
    
    # Load dialogue statistics
    stats_file = stats_dir / 'iemocap_statistics_report.json'
    if stats_file.exists():
        with open(stats_file) as f:
            results['dialogue_stats'] = json.load(f)
        print("✅ Loaded dialogue statistics")
    else:
        print(f"⚠️ Missing dialogue statistics: {stats_file}")
        results['dialogue_stats'] = None
    
    # Load discourse marker references
    ref_file = markers_dir / 'discourse_marker_references_validated.json'
    if ref_file.exists():
        with open(ref_file) as f:
            results['references'] = json.load(f)
        print("✅ Loaded reference validation")
    else:
        print(f"⚠️ Missing reference validation: {ref_file}")
        results['references'] = None
    
    # Load discourse marker statistics
    marker_file = markers_dir / 'discourse_marker_statistics.json'
    if marker_file.exists():
        with open(marker_file) as f:
            results['marker_stats'] = json.load(f)
        print("✅ Loaded marker statistics")
    else:
        print(f"⚠️ Missing marker statistics: {marker_file}")
        results['marker_stats'] = None
    
    return results

def generate_markdown_report(results):
    """Generate comprehensive markdown report"""
    
    report = f"""# IEMOCAP Statistical Analysis: Comprehensive Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

This report provides comprehensive statistics for IEMOCAP dataset analysis,
including dialogue length distributions, discourse marker patterns, and
experimental design recommendations for the TMLR paper.

---

"""
    
    # Dialogue Statistics Section
    if results['dialogue_stats']:
        stats_4 = results['dialogue_stats']['4way']['dialogue_stats']
        stats_6 = results['dialogue_stats']['6way']['dialogue_stats']
        
        report += """## 1. Dialogue Length Statistics

### Overall Statistics

| Metric | 4-way | 6-way |
|--------|-------|-------|
"""
        report += f"| Total Dialogues | {stats_4['total_dialogues']} | {stats_6['total_dialogues']} |\n"
        report += f"| Total Utterances | {stats_4['total_utterances']} | {stats_6['total_utterances']} |\n"
        report += f"| Mean Length | {stats_4['mean']:.1f} turns | {stats_6['mean']:.1f} turns |\n"
        report += f"| Median Length | {stats_4['median']:.1f} turns | {stats_6['median']:.1f} turns |\n"
        report += f"| **p90** | **{stats_4.get('p90', 0):.0f} turns** | **{stats_6.get('p90', 0):.0f} turns** |\n"
        report += f"| **p95 ⭐** | **{stats_4.get('p95', 0):.0f} turns** | **{stats_6.get('p95', 0):.0f} turns** |\n"
        report += f"| p99 | {stats_4.get('p99', 0):.0f} turns | {stats_6.get('p99', 0):.0f} turns |\n"
        report += f"| Max | {stats_4['max']} turns | {stats_6['max']} turns |\n\n"
        
        if 'coverage' in stats_4 and 'coverage' in stats_6:
            report += """### K-Coverage Analysis

| K Value | 4-way Coverage | 6-way Coverage |
|---------|---------------|---------------|
"""
            for k in [50, 70, 100, 150, 200]:
                cov4 = stats_4['coverage'].get(f'K{k}', 0) * 100
                cov6 = stats_6['coverage'].get(f'K{k}', 0) * 100
                report += f"| K={k} | {cov4:.1f}% | {cov6:.1f}% |\n"
            
            k100_cov = min(stats_4['coverage']['K100'], stats_6['coverage']['K100']) * 100
            p95_max = max(stats_4.get('p95', 0), stats_6.get('p95', 0))
            
            report += f"""
**⚠️ Key Finding**: K=100 covers only **{k100_cov:.1f}%** of dialogues.
With p95 = **{p95_max:.0f} turns**, consider extending K range for better coverage.

"""
    
    # Discourse Marker Statistics
    if results['marker_stats']:
        stats_4m = results['marker_stats']['4way']
        stats_6m = results['marker_stats']['6way']
        
        report += """---

## 2. Discourse Marker Statistics

### Summary

| Metric | 4-way | 6-way |
|--------|-------|-------|
"""
        report += f"| Total Markers | {stats_4m['total_markers']} | {stats_6m['total_markers']} |\n"
        report += f"| Unique Markers | {stats_4m['unique_markers']} | {stats_6m['unique_markers']} |\n"
        report += f"| Mean Position | {stats_4m['mean_position']:.3f} | {stats_6m['mean_position']:.3f} |\n"
        report += f"| Left Periphery (<0.1) | {stats_4m['left_periphery']*100:.1f}% | {stats_6m['left_periphery']*100:.1f}% |\n"
        report += f"| Right Periphery (>0.9) | {stats_4m['right_periphery']*100:.1f}% | {stats_6m['right_periphery']*100:.1f}% |\n"
        report += f"| **L/R Ratio ⭐** | **{stats_4m['lr_ratio']:.2f}** | **{stats_6m['lr_ratio']:.2f}** |\n\n"
        
        report += """### Top 10 Markers (4-way)

| Marker | Count | Percentage | Mean Position | L/R Ratio |
|--------|-------|-----------|---------------|-----------|
"""
        for marker, count in list(stats_4m['marker_frequencies'].items())[:10]:
            pct = count / stats_4m['total_markers'] * 100
            pos_info = stats_4m['marker_positions'][marker]
            report += f"| {marker} | {count} | {pct:.1f}% | {pos_info['mean_position']:.2f} | {pos_info['lr_ratio']:.2f} |\n"
    
    # K-Sweep Recommendations
    report += """
---

## 3. K-Sweep Range Recommendations

### Options

"""
    
    if results['dialogue_stats'] and 'coverage' in stats_4:
        recs = results['dialogue_stats']['recommendations']
        report += f"""| Option | K Range | Step | Rationale |
|--------|---------|------|-----------|
| **Conservative** | 0-100 | 5 | Paper reproduction (67.6% coverage) |
| **Data-driven** | {recs.get('k_sweep_p95_based', '0-205')} | 5 | p95-based (95% coverage) |
| **Comprehensive** | {recs.get('k_sweep_comprehensive', '0-270')} | 10 | Full coverage (100%) |

### Recommendation

1. **Primary**: K=0~100 (step=5, 21 values) for paper reproduction
2. **Extended**: K=0~{int(max(stats_4.get('p95', 195), stats_6.get('p95', 195))+10)} for 95% coverage
3. **Analysis**: Compare performance vs dialogue length

**Total runs**: 3 configs × 2 tasks × 5 seeds × 21 K-values = **630 training runs**

"""
    else:
        report += """| Option | K Range | Step | Rationale |
|--------|---------|------|-----------|
| **Conservative** | 0-100 | 5 | Paper default |
| **Data-driven** | 0-200 | 5 | Extended coverage |

Run dialogue statistics for detailed recommendations.

"""
    
    report += """---

## 4. Expected Results

### Table 5: Component Ablation (4-way)

| Configuration | Expected F1 | Expected ΔF1 |
|---------------|-------------|--------------|
| Full model | 80.68% | -- |
| - Position pooling | 78.66% | -2.02% |
| - Lexical fusion | 79.21% | -1.47% |
| - Context (K=0) | 60.32% | -20.36% |

### Table 3: Context Inversion

| Emotion | 4-way p90 | 6-way p90 | Shift |
|---------|-----------|-----------|-------|
| Angry | 64.00% | 24.50% | -39.5% ⚠️ |
| Sad | 29.00% | 62.50% | +33.5% ⚠️ |

---

## 5. Next Steps

1. ✅ Review this report
2. ⏳ Generate missing embeddings (if any)
3. ⏳ Run K-sweep experiments (630 runs, ~10-12 hours)
4. ⏳ Analyze results
5. ⏳ Generate figures

---

**Report End**
"""
    
    return report

def generate_latex_tables(results):
    """Generate LaTeX tables"""
    
    latex = r"""% LaTeX Tables for IEMOCAP Analysis

% Table: Dialogue Length Statistics
\begin{table}[h]
\centering
\caption{IEMOCAP Dialogue Length Statistics}
\label{tab:dialogue_stats}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{4-way} & \textbf{6-way} \\
\midrule
"""
    
    if results['dialogue_stats']:
        stats_4 = results['dialogue_stats']['4way']['dialogue_stats']
        stats_6 = results['dialogue_stats']['6way']['dialogue_stats']
        
        latex += f"Total Dialogues & {stats_4['total_dialogues']} & {stats_6['total_dialogues']} \\\\\n"
        latex += f"Total Utterances & {stats_4['total_utterances']} & {stats_6['total_utterances']} \\\\\n"
        latex += f"Mean Length & {stats_4['mean']:.1f} & {stats_6['mean']:.1f} \\\\\n"
        latex += f"Median & {stats_4['median']:.1f} & {stats_6['median']:.1f} \\\\\n"
        latex += f"$p_{{95}}$ & \\textbf{{{stats_4.get('p95', 0):.0f}}} & \\textbf{{{stats_6.get('p95', 0):.0f}}} \\\\\n"
        latex += f"Max & {stats_4['max']} & {stats_6['max']} \\\\\n"
        
        if 'coverage' in stats_4 and 'coverage' in stats_6:
            k100_4 = stats_4['coverage']['K100'] * 100
            k100_6 = stats_6['coverage']['K100'] * 100
            latex += f"Coverage@$K=100$ & {k100_4:.1f}\\% & {k100_6:.1f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Discourse Marker Table
    if results['marker_stats']:
        stats_4m = results['marker_stats']['4way']
        
        latex += r"""% Table: Top Discourse Markers
\begin{table}[h]
\centering
\caption{Top 10 Discourse Markers (4-way)}
\label{tab:markers}
\begin{tabular}{lccc}
\toprule
\textbf{Marker} & \textbf{Count} & \textbf{Position} & \textbf{L/R} \\
\midrule
"""
        
        for marker, count in list(stats_4m['marker_frequencies'].items())[:10]:
            pos_info = stats_4m['marker_positions'][marker]
            latex += f"{marker} & {count} & {pos_info['mean_position']:.2f} & {pos_info['lr_ratio']:.2f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex

def main():
    print("="*70)
    print("COMPREHENSIVE STATISTICAL REPORT GENERATOR")
    print("="*70)
    print(f"\n🏠 HOME: {HOME}")
    print(f"📁 STATS_DIR: {STATS_DIR}")
    print(f"📁 MARKERS_DIR: {MARKERS_DIR}")
    
    # Load results
    print("\n📂 Loading analysis results...")
    results = load_analysis_results()
    
    # Generate reports
    print("📝 Generating comprehensive report...")
    report = generate_markdown_report(results)
    
    print("📄 Generating LaTeX tables...")
    latex = generate_latex_tables(results)
    
    # Save outputs
    out_dir = HOME / "results" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'COMPREHENSIVE_REPORT.md', 'w') as f:
        f.write(report)
    print(f"✅ Saved: {out_dir}/COMPREHENSIVE_REPORT.md")
    
    with open(out_dir / 'latex_tables.tex', 'w') as f:
        f.write(latex)
    print(f"✅ Saved: {out_dir}/latex_tables.tex")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EXECUTIVE SUMMARY")
    print("="*70)
    
    if results['dialogue_stats']:
        stats_4 = results['dialogue_stats']['4way']['dialogue_stats']
        stats_6 = results['dialogue_stats']['6way']['dialogue_stats']
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   - p95 (4-way): {stats_4.get('p95', 0):.0f} turns")
        print(f"   - p95 (6-way): {stats_6.get('p95', 0):.0f} turns")
        if 'coverage' in stats_4:
            print(f"   - K=100 coverage: {min(stats_4['coverage']['K100'], stats_6['coverage']['K100'])*100:.1f}%")
    
    if results['marker_stats']:
        stats_4m = results['marker_stats']['4way']
        print(f"\n📌 DISCOURSE MARKERS:")
        print(f"   - Total markers: {stats_4m['total_markers']}")
        print(f"   - L/R ratio: {stats_4m['lr_ratio']:.2f}")
    
    print("\n✅ Report generation complete!")
    print(f"\n📖 Read: {out_dir}/COMPREHENSIVE_REPORT.md")

if __name__ == "__main__":
    main()