"""
Gated Fusion Model for Multimodal Emotion Recognition
- Text: Sentence-RoBERTa hierarchical embeddings (N, S_max, 768)
- Audio: OpenSMILE eGeMAPSv02 features (N, 88)
- Gating: learnable gate(s) to fuse modalities

Example usage:
python3 scripts/models/opensmile_sroberta/gated_fusion.py \
    --gate_type mlp \
    --hidden_dim 256 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3

"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class GatedFusionDataset(Dataset):
    """Load paired text and audio embeddings."""
    
    def __init__(
        self,
        text_path: Path,
        audio_path: Path,
        labels_path: Optional[Path] = None
    ):
        """
        Args:
            text_path: NPZ file with keys 'embeddings' (N, S_max, 768), 'lengths' (N,)
            audio_path: NPY file (N, 88)
            labels_path: Optional NPY file with labels (N,)

        """
        # Load text embeddings
        text_data = np.load(text_path, allow_pickle=True)
        self.text_embeddings = text_data['embeddings'].astype(np.float32)  # (N, S_max, 768)
        self.text_lengths = text_data['lengths'].astype(np.int32)  # (N,)
        self.N = self.text_embeddings.shape[0]
        self.S_max = self.text_embeddings.shape[1]
        self.text_dim = self.text_embeddings.shape[2]
        
        # Load audio embeddings
        self.audio_embeddings = np.load(audio_path).astype(np.float32)  # (N, 88)
        assert self.audio_embeddings.shape[0] == self.N, \
            f"Mismatch: text N={self.N}, audio N={self.audio_embeddings.shape[0]}"
        self.audio_dim = self.audio_embeddings.shape[1]
        
        # Load labels if provided
        self.labels = None
        if labels_path and Path(labels_path).exists():
            self.labels = np.load(labels_path).astype(np.int64)
            assert self.labels.shape[0] == self.N
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        text_emb = torch.from_numpy(self.text_embeddings[idx])  # (S_max, 768)
        text_len = torch.tensor(self.text_lengths[idx], dtype=torch.long)
        audio_emb = torch.from_numpy(self.audio_embeddings[idx])  # (88,)
        
        item = {
            'text': text_emb,
            'text_len': text_len,
            'audio': audio_emb,
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def collate_batch(batch):
    """Custom collate for variable-length text sequences."""
    keys = batch[0].keys()
    output = {}
    
    for key in keys:
        if key == 'text':
            # Stack with padding (all already padded to S_max in dataset)
            output[key] = torch.stack([item[key] for item in batch])  # (B, S_max, 768)
        elif key == 'text_len':
            output[key] = torch.stack([item[key] for item in batch])  # (B,)
        elif key == 'audio':
            output[key] = torch.stack([item[key] for item in batch])  # (B, 88)
        elif key == 'label':
            output[key] = torch.stack([item[key] for item in batch])  # (B,)
    
    return output


class GatedFusionModule(nn.Module):
    """
    Gated Fusion: gate * h_text + (1 - gate) * h_audio
    
    Both modalities projected to hidden dimension, then fused with learnable gate.

    """
    
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 88,
        hidden_dim: int = 256,
        num_classes: int = 4,
        gate_type: str = "scalar",  # "scalar", "vector", "mlp"
        dropout: float = 0.3,
        use_layer_norm: bool = True
    ):
        """
        Args:
            text_dim: Text embedding dimension (per token)
            audio_dim: Audio feature dimension
            hidden_dim: Projection/fusion dimension
            num_classes: Number of emotion classes
            gate_type: Type of gating mechanism
            dropout: Dropout rate
            use_layer_norm: Whether to apply layer normalization

        """
        super().__init__()
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.dropout_p = dropout
        
        # Project audio to hidden dimension
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project text to hidden dimension (applied per token, then pooled)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        if gate_type == "scalar":
            # Single scalar gate for all features
            self.gate = nn.Parameter(torch.tensor(0.5))
        elif gate_type == "vector":
            # Per-feature gates
            self.gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        elif gate_type == "mlp":
            # MLP-based gates: take concatenated features -> gate vector
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
        
        # Layer norm (optional)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: (B, S_max, 768) text embeddings
            text_len: (B,) actual sequence lengths
            audio: (B, 88) audio features
        
        Returns:
            logits: (B, num_classes) classification logits
            fused: (B, hidden_dim) fused representation

        """
        batch_size = text.shape[0]
        
        # Project audio
        audio_proj = self.audio_proj(audio)  # (B, hidden_dim)
        
        # Project and pool text
        text_proj = self.text_proj(text)  # (B, S_max, hidden_dim)
        
        # Masked mean pooling of text
        mask = torch.arange(text.shape[1], device=text.device).unsqueeze(0)  # (1, S_max)
        mask = mask < text_len.unsqueeze(1)  # (B, S_max)
        mask_float = mask.float().unsqueeze(-1)  # (B, S_max, 1)
        
        text_pooled = (text_proj * mask_float).sum(dim=1)  # (B, hidden_dim)
        text_pooled = text_pooled / mask_float.sum(dim=1).clamp_min(1e-6)  # (B, hidden_dim)
        
        # Compute gate
        if self.gate_type == "scalar":
            gate = torch.sigmoid(self.gate)  # scalar in [0, 1]
            fused = gate * text_pooled + (1 - gate) * audio_proj
        elif self.gate_type == "vector":
            gate = torch.sigmoid(self.gate)  # (hidden_dim,) in [0, 1]
            fused = gate * text_pooled + (1 - gate) * audio_proj
        elif self.gate_type == "mlp":
            concat = torch.cat([text_pooled, audio_proj], dim=1)  # (B, 2*hidden_dim)
            gate = self.gate_mlp(concat)  # (B, hidden_dim) in [0, 1]
            fused = gate * text_pooled + (1 - gate) * audio_proj
        
        # Optional layer norm
        if self.use_layer_norm:
            fused = self.fusion_norm(fused)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits, fused


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0
    total_samples = 0
    
    for batch in loader:
        text = batch['text'].to(device)
        text_len = batch['text_len'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(text, text_len, audio)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.shape[0]
        preds = logits.argmax(dim=1)
        total_acc += (preds == labels).sum().item()
        total_samples += labels.shape[0]
    
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0.0
    total_acc = 0
    total_samples = 0
    
    for batch in loader:
        text = batch['text'].to(device)
        text_len = batch['text_len'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        logits, _ = model(text, text_len, audio)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.shape[0]
        preds = logits.argmax(dim=1)
        total_acc += (preds == labels).sum().item()
        total_samples += labels.shape[0]
    
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(
        description="Gated Fusion for Multimodal Emotion Recognition"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/luay5/SenticCrystal/data"),
        help="Root data directory"
    )
    parser.add_argument(
        "--text_root",
        type=str,
        default="embeddings/4way/sroberta/sroberta-hier-avglast4-mean",
        help="Relative path to text embeddings"
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="embeddings/4way/opensmile/eGeMAPSv02",
        help="Relative path to audio embeddings"
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("/home/luay5/SenticCrystal/data/iemocap_4way_data"),
        help="Directory containing label CSVs"
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gate_type", type=str, default="mlp", 
                       choices=["scalar", "vector", "mlp"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    data_dir = args.data_dir
    text_dir = data_dir / args.text_root
    audio_dir = data_dir / args.audio_root
    
    print(f"[INFO] Text embeddings: {text_dir}")
    print(f"[INFO] Audio embeddings: {audio_dir}")
    
    # Load datasets
    print("[LOAD] Train split...")
    train_dataset = GatedFusionDataset(
        text_dir / "train_filtered.npz",
        audio_dir / "train_unified_filtered.npy"
    )
    
    print("[LOAD] Val split...")
    val_dataset = GatedFusionDataset(
        text_dir / "val_filtered.npz",
        audio_dir / "val_unified_filtered.npy"
    )
    
    print("[LOAD] Test split...")
    test_dataset = GatedFusionDataset(
        text_dir / "test_filtered.npz",
        audio_dir / "test_unified_filtered.npy"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Initialize model
    model = GatedFusionModule(
        text_dim=train_dataset.text_dim,
        audio_dim=train_dataset.audio_dim,
        hidden_dim=args.hidden_dim,
        num_classes=4,
        gate_type=args.gate_type,
        dropout=args.dropout
    ).to(device)
    
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    best_val_acc = 0.0
    best_model_path = Path("./best_gated_fusion.pt")
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Best model saved (acc={val_acc:.4f})")
    
    # Test evaluation
    print("\n[TEST] Evaluating best model...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
