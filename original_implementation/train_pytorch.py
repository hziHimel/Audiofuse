"""
AudioFuse — PyTorch Reimplementation
Faithful port of the TF/Keras dual-branch ViT + 1D-CNN architecture.

Usage:
    python train_pytorch.py --train_csv data/train.csv --val_csv data/val.csv \
                             --seed 42 --output_dir outputs/pytorch/
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import librosa
from sklearn.utils import class_weight
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
)
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────
class Config:
    IMG_SIZE = 224
    IN_CHANS = 1
    PATCH_SIZE = 16
    PROJ_DIM = 192
    NUM_HEADS = 8
    TRANSFORMER_LAYERS = 6
    MLP_RATIO = 2
    DROPOUT = 0.2

    SR = 22050
    WAV_SECONDS = 5
    WAV_LEN = SR * WAV_SECONDS   # 110 250

    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15

C = Config()
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ── Dataset ─────────────────────────────────────────────────────────────────
class PCGDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Spectrogram: load .npy, take channel 0
        npy = np.load(row["npy_filepath"])[:, :, 0].astype(np.float32)  # (H, W)
        spec = torch.from_numpy(npy).unsqueeze(0)  # (1, H, W)

        # Raw waveform
        wav, _ = librosa.load(row["filepath"], sr=C.SR, mono=True)
        if len(wav) > C.WAV_LEN:
            wav = wav[:C.WAV_LEN]
        else:
            wav = np.pad(wav, (0, C.WAV_LEN - len(wav)), "constant")
        wave = torch.from_numpy(wav.astype(np.float32))  # (WAV_LEN,)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return spec, wave, label


# ── Model ───────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, proj_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, proj_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # (B, proj_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpectrogramViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(C.PATCH_SIZE, C.IN_CHANS, C.PROJ_DIM)
        num_patches = (C.IMG_SIZE // C.PATCH_SIZE) ** 2
        self.pos_emb = nn.Embedding(num_patches, C.PROJ_DIM)
        self.blocks = nn.Sequential(
            *[TransformerBlock(C.PROJ_DIM, C.NUM_HEADS, C.MLP_RATIO, C.DROPOUT)
              for _ in range(C.TRANSFORMER_LAYERS)]
        )
        self.norm = nn.LayerNorm(C.PROJ_DIM, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)                        # (B, N, 192)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)                           # (B, 192)


class WaveformCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(256, 64)

    def forward(self, x):
        x = x.unsqueeze(1)           # (B, 1, WAV_LEN)
        x = self.layers(x)           # (B, 256, T')
        x = x.mean(dim=-1)           # (B, 256)
        return F.relu(self.fc(x))    # (B, 64)


class AudioFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_branch = SpectrogramViT()
        self.wave_branch = WaveformCNN()
        self.head = nn.Sequential(
            nn.Linear(192 + 64, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 1),
        )

    def forward(self, spec, wave):
        f_spec = self.spec_branch(spec)    # (B, 192)
        f_wave = self.wave_branch(wave)    # (B, 64)
        fused = torch.cat([f_spec, f_wave], dim=1)  # (B, 256)
        return self.head(fused).squeeze(1)  # (B,) logits


# ── Training helpers ────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, pos_weight, train=True):
    model.train(train)
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for spec, wave, labels in tqdm(loader, leave=False):
            spec = spec.to(DEVICE)
            wave = wave.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(spec, wave)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_ds = PCGDataset(train_df)
    val_ds = PCGDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Class-weighted positive weight for BCEWithLogitsLoss
    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.4f}")

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10,
                                  min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_seed{seed}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, pos_weight, train=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, optimizer, pos_weight, train=False)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch:3d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} auc={vl_auc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= C.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Final evaluation with best weights ─────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for spec, wave, labels in val_loader:
            logits = model(spec.to(DEVICE), wave.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # Save per-sample predictions for threshold analysis
    preds_path = os.path.join(args.output_dir, f"val_preds_seed{seed}.csv")
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(preds_path, index=False)

    # Fixed-threshold evaluation (0.5)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Threshold sweep: find the threshold that maximises F1
    opt_thresh, opt_f1 = sweep_threshold(y_true, y_prob)
    y_pred_opt = (y_prob > opt_thresh).astype(int)
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_mcc = matthews_corrcoef(y_true, y_pred_opt)

    print(f"\n{'='*50}")
    print(f"Seed {seed} Results (threshold=0.50):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"Optimal threshold = {opt_thresh:.2f}  (val F1={opt_f1:.4f})")
    print(f"  Accuracy : {opt_acc:.4f}")
    print(f"  F1-Score : {opt_f1:.4f}")
    print(f"  MCC      : {opt_mcc:.4f}")
    print(f"{'='*50}")
    return {
        "seed": seed,
        "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc,
        "opt_threshold": opt_thresh,
        "opt_accuracy": opt_acc, "opt_f1": opt_f1, "opt_mcc": opt_mcc,
    }


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    lo: float = 0.1, hi: float = 0.9, step: float = 0.05
                    ) -> tuple[float, float]:
    """Return (threshold, f1) maximising F1 on the given predictions."""
    thresholds = np.arange(lo, hi + step / 2, step)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh), float(best_f1)


# ── Entry point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"# Training with seed={seed}")
        print(f"{'#'*60}")
        results = train_one_seed(args, seed)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    print("\n\nFinal Summary (mean ± std over seeds):")
    for col in ["accuracy", "f1", "roc_auc", "mcc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/results.csv")


if __name__ == "__main__":
    main()
