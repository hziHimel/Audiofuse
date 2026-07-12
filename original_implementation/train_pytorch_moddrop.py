"""
AudioFuse — Modality Dropout Training (Direction 3.1, gradient-dominance remedy)

Trains AudioFuse from random init with per-sample modality dropout: each training
sample independently has (prob p) one modality zeroed, forcing both branches to be
independently useful. Samples that lose the waveform force ViT-only classification.

Protocol matches the random-init baseline (shuffle, pos_weight, LR=3e-4,
plateau-on-loss, checkpoint-on-val-acc, patience=15) — the only difference is
modality dropout on training batches. Val is never dropped. Logs per-branch
gradient norms for consistency with the other remedy runs.

Usage:
    python train_pytorch_moddrop.py --train_csv data/train.csv --val_csv data/val.csv \
        --p 0.5 --seed 1 --output_dir outputs/pytorch_moddrop/
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

from train_pytorch import Config, AudioFuse, PCGDataset, sweep_threshold, DEVICE
from grad_flow import branch_grad_norm
from modality_dropout import apply_modality_dropout

C = Config()


def run_train_epoch(model, loader, optimizer, pos_weight, p, rec, epoch):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for spec, wave, labels in tqdm(loader, leave=False):
        spec, wave, labels = spec.to(DEVICE), wave.to(DEVICE), labels.to(DEVICE)
        spec, wave = apply_modality_dropout(spec, wave, p)

        logits = model(spec, wave)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        rec.append({
            "epoch": epoch,
            "grad_norm_spec": branch_grad_norm(model.spec_branch),
            "grad_norm_wave": branch_grad_norm(model.wave_branch),
        })
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, auc


@torch.no_grad()
def run_val_epoch(model, loader, pos_weight):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for spec, wave, labels in loader:
        spec, wave, labels = spec.to(DEVICE), wave.to(DEVICE), labels.to(DEVICE)
        logits = model(spec, wave)                     # val: no dropout
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        total_loss += loss.item() * len(labels)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  default="data/train.csv")
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_moddrop/")
    parser.add_argument("--p", type=float, default=0.5, help="per-sample modality dropout prob")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    train_loader = DataLoader(PCGDataset(train_df), batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight={pos_weight.item():.4f}  p={args.p}  [modality dropout]")

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_ckpt = os.path.join(args.output_dir, f"best_seed{args.seed}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0
    grad_rec, epoch_records = [], []

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_train_epoch(
            model, train_loader, optimizer, pos_weight, args.p, grad_rec, epoch)
        vl_loss, vl_acc, vl_auc = run_val_epoch(model, val_loader, pos_weight)
        scheduler.step(vl_loss)

        ep = [r for r in grad_rec if r["epoch"] == epoch]
        mean_spec = np.mean([r["grad_norm_spec"] for r in ep])
        mean_wave = np.mean([r["grad_norm_wave"] for r in ep])
        ratio = mean_wave / (mean_spec + 1e-12)
        epoch_records.append({
            "epoch": epoch, "val_auc": vl_auc, "val_acc": vl_acc,
            "grad_norm_spec": mean_spec, "grad_norm_wave": mean_wave,
            "grad_ratio_wave_over_spec": ratio,
        })

        print(f"Epoch {epoch:3d} | train auc={tr_auc:.4f} | val auc={vl_auc:.4f} | "
              f"grad_norm wave={mean_wave:.4f} spec={mean_spec:.4f} ratio(w/s)={ratio:.2f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= C.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(epoch_records).to_csv(
        os.path.join(args.output_dir, "moddrop_epoch.csv"), index=False)

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for spec, wave, labels in val_loader:
            logits = model(spec.to(DEVICE), wave.to(DEVICE))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_prob = np.array(all_labels), np.array(all_probs)
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(
        os.path.join(args.output_dir, f"val_preds_seed{args.seed}.csv"), index=False)

    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    opt_thresh, opt_f1 = sweep_threshold(y_true, y_prob)
    y_pred_opt = (y_prob > opt_thresh).astype(int)
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_mcc = matthews_corrcoef(y_true, y_pred_opt)

    print(f"\n{'='*50}")
    print(f"Seed {args.seed} Results [MODALITY-DROPOUT p={args.p}] (threshold=0.50):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"Optimal threshold = {opt_thresh:.2f}  (val F1={opt_f1:.4f})")
    print(f"  Accuracy : {opt_acc:.4f}  F1: {opt_f1:.4f}  MCC: {opt_mcc:.4f}")
    print(f"{'='*50}")
    print(f"Checkpoint: {best_ckpt}  |  Grad log: moddrop_epoch.csv")


if __name__ == "__main__":
    main()
