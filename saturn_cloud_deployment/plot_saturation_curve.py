"""
Real-time Saturation Curve Plotter
==================================

Plot accuracy and F1 curves as experiments complete to visualize saturation point.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import glob
import numpy as np
from pathlib import Path
import time

def plot_saturation_curves(results_dir="results/turn_experiments/", save_path="saturation_curves.png"):
    """Plot saturation curves for K=0-50 experiments."""
    
    # Collect all config146 results
    result_files = glob.glob(f"{results_dir}/config146_k*_results.json")
    
    data = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Extract K value from filename
            filename = Path(file_path).stem
            k_value = int(filename.split('_k')[1].split('_')[0])
            model = filename.split('_')[-2]  # lstm or mlp
            
            metrics = result.get('test_results', {})
            data.append({
                'K': k_value,
                'Model': model.upper(),
                'Accuracy': metrics.get('accuracy', 0),
                'Macro_F1': metrics.get('macro_f1', 0),
                'Weighted_F1': metrics.get('weighted_f1', 0)
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not data:
        print("No results found!")
        return
    
    df = pd.DataFrame(data)
    df = df.sort_values('K')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SenticCrystal Context Window Saturation Analysis (K=0-50)', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy curves
    ax1 = axes[0, 0]
    for model in ['LSTM', 'MLP']:
        model_data = df[df['Model'] == model]
        ax1.plot(model_data['K'], model_data['Accuracy'], marker='o', label=f'{model} Accuracy', linewidth=2)
    ax1.set_xlabel('Context Window Size (K)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Context Window Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Macro F1 curves  
    ax2 = axes[0, 1]
    for model in ['LSTM', 'MLP']:
        model_data = df[df['Model'] == model]
        ax2.plot(model_data['K'], model_data['Macro_F1'], marker='s', label=f'{model} Macro-F1', linewidth=2)
    ax2.set_xlabel('Context Window Size (K)')
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Macro F1 vs Context Window Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weighted F1 curves
    ax3 = axes[1, 0]
    for model in ['LSTM', 'MLP']:
        model_data = df[df['Model'] == model]
        ax3.plot(model_data['K'], model_data['Weighted_F1'], marker='^', label=f'{model} Weighted-F1', linewidth=2)
    ax3.set_xlabel('Context Window Size (K)')
    ax3.set_ylabel('Weighted F1')
    ax3.set_title('Weighted F1 vs Context Window Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find best performances
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_macro = df.loc[df['Macro_F1'].idxmax()]
    best_weighted = df.loc[df['Weighted_F1'].idxmax()]
    
    summary_text = f"""
    📊 PERFORMANCE SUMMARY
    
    🏆 Best Accuracy: {best_acc['Accuracy']:.4f}
       K={best_acc['K']}, {best_acc['Model']}
    
    🎯 Best Macro-F1: {best_macro['Macro_F1']:.4f}
       K={best_macro['K']}, {best_macro['Model']}
    
    ⚖️ Best Weighted-F1: {best_weighted['Weighted_F1']:.4f}
       K={best_weighted['K']}, {best_weighted['Model']}
    
    📈 Total Experiments: {len(df)}
    📋 K Range: {df['K'].min()}-{df['K'].max()}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saturation curve saved to: {save_path}")
    
    return df

def watch_experiments(results_dir="results/turn_experiments/", update_interval=30):
    """Watch for new results and update plot in real-time."""
    print("🔍 Watching for new experimental results...")
    print(f"📁 Directory: {results_dir}")
    print(f"⏱️ Update interval: {update_interval}s")
    
    while True:
        try:
            df = plot_saturation_curves(results_dir)
            if df is not None and not df.empty:
                completed_k_values = sorted(df['K'].unique())
                print(f"✅ Completed K values: {completed_k_values}")
                print(f"📊 Total experiments: {len(df)}")
            else:
                print("⏳ No results found yet...")
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time monitoring...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(update_interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot saturation curves for SenticCrystal experiments")
    parser.add_argument('--results_dir', default='results/turn_experiments/', help='Results directory')
    parser.add_argument('--watch', action='store_true', help='Watch for new results in real-time')
    parser.add_argument('--interval', type=int, default=30, help='Update interval for watching (seconds)')
    
    args = parser.parse_args()
    
    if args.watch:
        watch_experiments(args.results_dir, args.interval)
    else:
        plot_saturation_curves(args.results_dir)