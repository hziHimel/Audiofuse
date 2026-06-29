"""
AudioFuse — Focal Loss variant (Direction 1.1)

Replaces BCE + pos_weight with focal loss (alpha=0.25, gamma=2.0) to address
the 3.33:1 class imbalance without needing an explicit positive weight.

Usage:
    python train_pytorch_focal.py --train_csv data/train.csv --val_csv data/val.csv \
                                   --seed 1 --output_dir outputs/pytorch_focal/
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

# Reuse all model/dataset definitions from the base script
from train_pytorch import Config, PCGDataset, AudioFuse, sweep_threshold, DEVICE
from losses import focal_loss_with_logits

C = Config()

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def run_epoch(model, loader, optimizer, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for spec, wave, labels in tqdm(loader, leave=False):
            spec = spec.to(DEVICE)
            wave = wave.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(spec, wave)
            loss = focal_loss_with_logits(logits, labels, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


def train_one_seed(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_loader = DataLoader(PCGDataset(train_df), batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_seed{seed}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Resume from checkpoint if provided or if one already exists in output_dir
    resume_ckpt = args.resume_ckpt or (best_ckpt if os.path.exists(best_ckpt) else None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=DEVICE))
        print(f"Resumed from checkpoint: {resume_ckpt}")

    print(f"Focal loss: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, train=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, optimizer, train=False)
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

    preds_path = os.path.join(args.output_dir, f"val_preds_seed{seed}.csv")
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(preds_path, index=False)

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
    print(f"Seed {seed} Results [FOCAL loss] (threshold=0.50):")
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
    parser.add_argument("--output_dir", default="outputs/pytorch_focal/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--resume_ckpt", default=None,
                        help="Path to checkpoint to resume from. If omitted, auto-detects existing best_seedN.pt.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# Training with seed={seed} [FOCAL LOSS]\n{'#'*60}")
        all_results.append(train_one_seed(args, seed))

    results_df = pd.DataFrame(all_results)
    if len(all_results) > 1:
        print("\nFinal Summary (mean ± std over seeds):")
        for col in ["accuracy", "f1", "roc_auc", "mcc"]:
            print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    main()
