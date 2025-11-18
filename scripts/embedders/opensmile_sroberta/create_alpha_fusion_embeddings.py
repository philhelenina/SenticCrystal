#!/usr/bin/env python3
"""
create_alpha_fusion_embeddings.py
Generate α-weighted fusion embeddings for IEMOCAP 4-way
h_fused = (1-α) * H_norm + α * W_proj @ S_norm
Output: data/embeddings/4way/opensmile_sroberta/alpha_fusion/{005,010,020,030}/{last,avg_last4}/wmean_pos_rev/{split}.npz
"""
import numpy as np
from pathlib import Path

HOME = Path("/home/jovyan/workspace/SenticCrystal/")

def l2_normalize(x, axis=-1, eps=1e-12):
    """L2 normalization"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def create_projection_matrix(in_dim, out_dim, seed=42):
    """Create random projection matrix (out_dim, in_dim)"""
    np.random.seed(seed)
    W = np.random.randn(out_dim, in_dim).astype(np.float32)
    # Normalize each column
    for i in range(in_dim):
        W[:, i] = W[:, i] / (np.linalg.norm(W[:, i]) + 1e-8)
    return W  # (768, 88)

def fuse_with_alpha(H, S, alpha, W_proj):
    """α-weighted fusion"""
    # Normalize
    H_norm = l2_normalize(H, axis=1)
    S_norm = l2_normalize(S, axis=1)
    
    # Project: (N, 88) @ (88, 768)^T = (N, 768)
    S_proj = S_norm @ W_proj.T
    S_proj = l2_normalize(S_proj, axis=1)
    
    # Fusion
    h_fused = (1 - alpha) * H_norm + alpha * S_proj
    return h_fused

def main():
    print("="*70)
    print("Creating α-Fusion Embeddings for IEMOCAP 4-way")
    print("="*70)
    
    # Config
    TASK = "4way"
    LAYERS = ["avg_last4"]
    POOL = "wmean_pos_rev"
    ALPHAS = [0.05, 0.10, 0.20, 0.30]
    SPLITS = ["train", "val", "test"]
    
    print(f"\nConfig:")
    print(f"  Task: {TASK}")
    print(f"  Layers: {LAYERS}")
    print(f"  Pool: {POOL}")
    print(f"  Alphas: {ALPHAS}")
    
    # Create projection matrix (shared)
    print("\nCreating W_proj (768, 88)...")
    W_proj = create_projection_matrix(in_dim=88, out_dim=768, seed=42)
    
    # Process each layer
    for LAYER in LAYERS:
        print(f"\n{'='*70}")
        print(f"Layer: {LAYER}")
        print(f"{'='*70}")
        
        roberta_base = HOME / "data/embeddings" / TASK / "sroberta" / LAYER / POOL
        opensmile_base = HOME / "data/embeddings" / TASK / "opensmile/eGeMAPSv02"
        
        print(f"  SRoBERTa: {roberta_base}")
        print(f"  openSMILE: {opensmile_base}")
        
        for alpha in ALPHAS:
            alpha_tag = f"{int(alpha*100):03d}"
            out_base = HOME / "data/embeddings" / TASK / "opensmile_sroberta/alpha_fusion" / alpha_tag / LAYER / POOL
            out_base.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  α={alpha:.2f} (tag={alpha_tag})")
            
            for split in SPLITS:
                # Load
                H = np.load(roberta_base / f"{split}_filtered.npz")["embeddings"].astype(np.float32)
                S = np.load(opensmile_base / f"{split}_filtered.npy").astype(np.float32)
                S = np.squeeze(S, axis=1)
                
                # Align
                n = min(len(H), len(S))
                H, S = H[:n], S[:n]
                
                # Fuse
                h_fused = fuse_with_alpha(H, S, alpha, W_proj)
                
                # Save as .npz
                out_path = out_base / f"{split}.npz"
                np.savez_compressed(out_path, embeddings=h_fused)
                print(f"    {split}: {h_fused.shape} → {out_path.relative_to(HOME)}")
    
    print("\n" + "="*70)
    print("✓ All embeddings created!")
    print("="*70)
    print("\nOutput structure:")
    print("  data/embeddings/4way/opensmile_sroberta/alpha_fusion/")
    print("    ├── 005/")
    print("    │   └── avg_last4/wmean_pos_rev/{train,val,test}.npz")
    print("    ├── 010/")
    print("    │   └── avg_last4/wmean_pos_rev/{train,val,test}.npz")
    print("    ├── 020/")
    print("    │   └── avg_last4/wmean_pos_rev/{train,val,test}.npz")
    print("    └── 030/")
    print("        └── avg_last4/wmean_pos_rev/{train,val,test}.npz")

if __name__ == "__main__":
    main()