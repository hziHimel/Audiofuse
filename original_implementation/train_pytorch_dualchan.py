"""
AudioFuse — Dual-Channel Spectrogram Input (Direction 1.2)

Identical to the baseline except both .npy channels are used as input to the
ViT branch: channel 0 = log-Mel spectrogram, channel 1 = CWT scalogram.
IN_CHANS is set to 2 so the patch embedding conv accepts 2-channel input.
This matches the original authors' preprocessing intent (saved as H×W×2).

Usage:
    python train_pytorch_dualchan.py --train_csv data/train.csv \
        --val_csv data/val.csv --seeds 1 --output_dir outputs/pytorch_dualchan/
"""

import os
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from train_pytorch import Config, SpectrogramViT, WaveformCNN, sweep_threshold, DEVICE

C = Config()


class PCGDatasetDual(Dataset):
    """Like PCGDataset but loads both spectrogram channels (H, W, 2) → (2, H, W)."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy = np.load(row["npy_filepath"]).astype(np.float32)  # (H, W, 2)
        spec = torch.from_numpy(npy).permute(2, 0, 1)          # (2, H, W)

        wav, _ = librosa.load(row["filepath"], sr=C.SR, mono=True)
        if len(wav) > C.WAV_LEN:
            wav = wav[:C.WAV_LEN]
        else:
            wav = np.pad(wav, (0, C.WAV_LEN - len(wav)), "constant")
        wave = torch.from_numpy(wav.astype(np.float32))

        label = torch.tensor(row["label"], dtype=torch.float32)
        return spec, wave, label


class AudioFuseDual(nn.Module):
    """AudioFuse with 2-channel ViT branch (log-Mel + scalogram)."""
    def __init__(self):
        super().__init__()
        # Patch embed with IN_CHANS=2 instead of 1
        self.patch_embed = nn.Conv2d(2, C.PROJ_DIM, kernel_size=C.PATCH_SIZE, stride=C.PATCH_SIZE)
        num_patches = (C.IMG_SIZE // C.PATCH_SIZE) ** 2
        self.pos_emb = nn.Embedding(num_patches, C.PROJ_DIM)

        from train_pytorch import TransformerBlock
        self.blocks = nn.Sequential(
            *[TransformerBlock(C.PROJ_DIM, C.NUM_HEADS, C.MLP_RATIO, C.DROPOUT)
              for _ in range(C.TRANSFORMER_LAYERS)]
        )
        self.norm = nn.LayerNorm(C.PROJ_DIM, eps=1e-6)
        self.wave_branch = WaveformCNN()
        self.head = nn.Sequential(
            nn.Linear(192 + 64, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 1),
        )

    def forward(self, spec, wave):
        x = self.patch_embed(spec)                     # (B, 192, H/p, W/p)
        B, C_, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)               # (B, N, 192)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.norm(x)
        f_spec = x.mean(dim=1)                         # (B, 192)
        f_wave = self.wave_branch(wave)                # (B, 64)
        fused = torch.cat([f_spec, f_wave], dim=1)
        return self.head(fused).squeeze(1)


def run_epoch(model, loader, optimizer, pos_weight, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for spec, wave, labels in tqdm(loader, leave=False):
            spec, wave, labels = spec.to(DEVICE), wave.to(DEVICE), labels.to(DEVICE)
            logits = model(spec, wave)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, auc


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_loader = DataLoader(PCGDatasetDual(train_df), batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(PCGDatasetDual(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.4f}  [dual-channel: log-Mel + CWT scalogram]")

    model = AudioFuseDual().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_seed{seed}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        print(f"Resumed from checkpoint: {best_ckpt}")

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

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for spec, wave, labels in val_loader:
            logits = model(spec.to(DEVICE), wave.to(DEVICE))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(
        os.path.join(args.output_dir, f"val_preds_seed{seed}.csv"), index=False)

    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    opt_thresh, opt_f1 = sweep_threshold(y_true, y_prob)
    y_pred_opt = (y_prob > opt_thresh).astype(int)
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_mcc = matthews_corrcoef(y_true, y_pred_opt)

    print(f"\n{'='*50}")
    print(f"Seed {seed} Results [DUAL-CHANNEL] (threshold=0.50):")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_dualchan/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# Training with seed={seed} [DUAL-CHANNEL]\n{'#'*60}")
        all_results.append(train_one_seed(args, seed))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    if len(all_results) > 1:
        print("\nFinal Summary (mean ± std over seeds):")
        for col in ["accuracy", "f1", "roc_auc", "mcc"]:
            print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")


if __name__ == "__main__":
    main()
