"""
AudioFuse — Mixup Augmentation (Direction 1.1)

Identical to the baseline except training batches are mixed via Mixup (α=0.4).
Both spec and waveform inputs receive the same λ so the two modalities remain
consistent within each mixed sample. Mixup is only applied during training;
validation is identical to the baseline.

Usage:
    python train_pytorch_mixup.py --train_csv data/train.csv \
        --val_csv data/val.csv --seeds 1 --output_dir outputs/pytorch_mixup/
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from train_pytorch import Config, PCGDataset, AudioFuse, sweep_threshold, DEVICE

C = Config()
MIXUP_ALPHA = 0.4


def mixup_batch(spec, wave, labels, alpha: float = MIXUP_ALPHA):
    """Apply Mixup to a batch: same λ for spec and wave to keep modalities consistent."""
    lam = np.random.beta(alpha, alpha)
    B = spec.size(0)
    idx = torch.randperm(B, device=spec.device)
    spec_mix = lam * spec + (1 - lam) * spec[idx]
    wave_mix = lam * wave + (1 - lam) * wave[idx]
    labels_a, labels_b = labels, labels[idx]
    return spec_mix, wave_mix, labels_a, labels_b, lam


def run_epoch(model, loader, optimizer, pos_weight, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for spec, wave, labels in tqdm(loader, leave=False):
            spec = spec.to(DEVICE)
            wave = wave.to(DEVICE)
            labels = labels.to(DEVICE)

            if train:
                spec, wave, labels_a, labels_b, lam = mixup_batch(spec, wave, labels)
                logits = model(spec, wave)
                loss = (lam * F.binary_cross_entropy_with_logits(logits, labels_a, pos_weight=pos_weight)
                        + (1 - lam) * F.binary_cross_entropy_with_logits(logits, labels_b, pos_weight=pos_weight))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # For metric tracking use original labels (labels_a = pre-shuffle)
                eval_labels = labels_a
            else:
                logits = model(spec, wave)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                eval_labels = labels

            total_loss += loss.item() * len(eval_labels)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(eval_labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, auc


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_loader = DataLoader(PCGDataset(train_df), batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.4f}  mixup_alpha = {MIXUP_ALPHA}")

    model = AudioFuse().to(DEVICE)
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
    print(f"Seed {seed} Results [MIXUP α={MIXUP_ALPHA}] (threshold=0.50):")
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
    parser.add_argument("--output_dir", default="outputs/pytorch_mixup/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# Training with seed={seed} [MIXUP α={MIXUP_ALPHA}]\n{'#'*60}")
        all_results.append(train_one_seed(args, seed))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    if len(all_results) > 1:
        print("\nFinal Summary (mean ± std over seeds):")
        for col in ["accuracy", "f1", "roc_auc", "mcc"]:
            print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")


if __name__ == "__main__":
    main()
