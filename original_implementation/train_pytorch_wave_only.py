"""
AudioFuse — Waveform-Only Baseline (PyTorch)
Trains only the 1D-CNN branch on raw waveforms. No spectrogram input.

Usage:
    python train_pytorch_wave_only.py --train_csv data/train.csv \
                                       --val_csv data/val.csv \
                                       --output_dir outputs/pytorch_wave_only/ \
                                       --seeds 1
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


# ── Config ──────────────────────────────────────────────────────────────────
class Config:
    SR          = 22050
    WAV_SECONDS = 5
    WAV_LEN     = SR * WAV_SECONDS   # 110,250

    BATCH_SIZE   = 32
    EPOCHS       = 200
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 15

C = Config()
DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)


# ── Dataset ──────────────────────────────────────────────────────────────────
class WaveformDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, _ = librosa.load(row["filepath"], sr=C.SR, mono=True)
        if len(wav) > C.WAV_LEN:
            wav = wav[:C.WAV_LEN]
        else:
            wav = np.pad(wav, (0, C.WAV_LEN - len(wav)), "constant")
        wave  = torch.from_numpy(wav.astype(np.float32))   # (WAV_LEN,)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return wave, label


# ── Model ────────────────────────────────────────────────────────────────────
class WaveformCNN(nn.Module):
    """Waveform-only model: 1D-CNN branch + classification head."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64,  128, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # (B, 1, WAV_LEN)
        x = self.layers(x)      # (B, 256, T')
        x = x.mean(dim=-1)      # (B, 256)
        return self.head(x).squeeze(1)   # (B,) logits


# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, pos_weight, train=True):
    model.train(train)
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for wave, labels in tqdm(loader, leave=False):
            wave   = wave.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(wave)
            loss   = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    train_ds = WaveformDataset(train_df)
    val_ds   = WaveformDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False, num_workers=2)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.4f}")

    model     = WaveformCNN().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_seed{seed}.pt")
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, pos_weight, train=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader,   optimizer, pos_weight, train=False)
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

    # ── Final evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for wave, labels in val_loader:
            logits = model(wave.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Seed {seed} Results (Waveform-Only / 1D-CNN):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"{'='*50}")
    return {"seed": seed, "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc}


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  default="data/train.csv")
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_wave_only/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("Model: Waveform-Only (1D-CNN branch)")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"# Training with seed={seed}")
        print(f"{'#'*60}")
        all_results.append(train_one_seed(args, seed))

    results_df = pd.DataFrame(all_results)
    print("\n\nFinal Summary (mean ± std):")
    for col in ["accuracy", "f1", "roc_auc", "mcc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/results.csv")


if __name__ == "__main__":
    main()
