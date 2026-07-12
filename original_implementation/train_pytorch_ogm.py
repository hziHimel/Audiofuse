"""
AudioFuse — OGM-GE Training (Direction 3.1, gradient-dominance remedy)

Trains AudioFuse from random init with On-the-fly Gradient Modulation (OGM-GE,
Peng et al. CVPR 2022) to counteract the waveform CNN dominating gradients.

Protocol matches the random-init baseline (shuffle, pos_weight, LR=3e-4,
plateau-on-loss, checkpoint-on-val-acc, patience=15) so OGM-GE vs baseline is a
clean A/B — the ONLY difference is the gradient modulation. Logs per-branch
gradient norms BEFORE and AFTER modulation to show the rebalancing effect
(reuses grad_flow instrumentation).

Usage:
    python train_pytorch_ogm.py --train_csv data/train.csv --val_csv data/val.csv \
        --alpha 0.5 --seed 1 --output_dir outputs/pytorch_ogm/
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
from ogm import true_class_confidence, modulation_coeffs

C = Config()


@torch.no_grad()
def branch_confidences(model, spec, wave, labels):
    """Per-branch true-class confidence via zeroed-branch forward passes.
    Uses eval mode so BatchNorm running stats are not disturbed."""
    was_training = model.training
    model.eval()
    p_spec = torch.sigmoid(model(spec, torch.zeros_like(wave))).cpu().numpy()
    p_wave = torch.sigmoid(model(torch.zeros_like(spec), wave)).cpu().numpy()
    model.train(was_training)
    y = labels.cpu().numpy()
    return true_class_confidence(p_spec, y), true_class_confidence(p_wave, y)


def apply_ge_noise(param, sigma):
    """Generalization Enhancement: add zero-mean Gaussian noise scaled to the
    gradient's own std, restoring gradient dynamics lost to damping."""
    if sigma > 0 and param.grad is not None:
        param.grad += torch.randn_like(param.grad) * sigma * param.grad.std()


def run_train_epoch(model, loader, optimizer, pos_weight, alpha, ge_sigma, rec, epoch):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for spec, wave, labels in tqdm(loader, leave=False):
        spec, wave, labels = spec.to(DEVICE), wave.to(DEVICE), labels.to(DEVICE)
        logits = model(spec, wave)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()

        # pre-modulation gradient norms
        g_spec_pre = branch_grad_norm(model.spec_branch)
        g_wave_pre = branch_grad_norm(model.wave_branch)

        # estimate per-branch contribution and modulate dominant branch
        conf_spec, conf_wave = branch_confidences(model, spec, wave, labels)
        k_spec, k_wave = modulation_coeffs(conf_spec, conf_wave, alpha)
        for p in model.spec_branch.parameters():
            if p.grad is not None:
                p.grad *= k_spec
                apply_ge_noise(p, ge_sigma)
        for p in model.wave_branch.parameters():
            if p.grad is not None:
                p.grad *= k_wave
                apply_ge_noise(p, ge_sigma)

        # post-modulation gradient norms
        g_spec_post = branch_grad_norm(model.spec_branch)
        g_wave_post = branch_grad_norm(model.wave_branch)
        rec.append({
            "epoch": epoch, "k_spec": k_spec, "k_wave": k_wave,
            "ratio_pre": g_wave_pre / (g_spec_pre + 1e-12),
            "ratio_post": g_wave_post / (g_spec_post + 1e-12),
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
        logits = model(spec, wave)
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
    parser.add_argument("--output_dir", default="outputs/pytorch_ogm/")
    parser.add_argument("--alpha", type=float, default=0.5, help="OGM modulation strength")
    parser.add_argument("--ge_sigma", type=float, default=0.0, help="GE noise scale (0=off)")
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
    print(f"pos_weight={pos_weight.item():.4f}  alpha={args.alpha}  ge_sigma={args.ge_sigma}  [OGM-GE]")

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_ckpt = os.path.join(args.output_dir, f"best_seed{args.seed}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0
    mod_records, epoch_records = [], []

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_train_epoch(
            model, train_loader, optimizer, pos_weight, args.alpha, args.ge_sigma, mod_records, epoch)
        vl_loss, vl_acc, vl_auc = run_val_epoch(model, val_loader, pos_weight)
        scheduler.step(vl_loss)

        ep = [r for r in mod_records if r["epoch"] == epoch]
        mean_pre  = np.mean([r["ratio_pre"] for r in ep])
        mean_post = np.mean([r["ratio_post"] for r in ep])
        mean_kwave = np.mean([r["k_wave"] for r in ep])
        mean_kspec = np.mean([r["k_spec"] for r in ep])
        epoch_records.append({
            "epoch": epoch, "val_auc": vl_auc, "val_acc": vl_acc,
            "ratio_pre": mean_pre, "ratio_post": mean_post,
            "k_spec": mean_kspec, "k_wave": mean_kwave,
        })

        print(f"Epoch {epoch:3d} | train auc={tr_auc:.4f} | val auc={vl_auc:.4f} | "
              f"grad ratio pre={mean_pre:.2f} post={mean_post:.2f} | "
              f"k_wave={mean_kwave:.3f} k_spec={mean_kspec:.3f}")

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
        os.path.join(args.output_dir, "ogm_epoch.csv"), index=False)

    # final eval on best checkpoint
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
    print(f"Seed {args.seed} Results [OGM-GE alpha={args.alpha}] (threshold=0.50):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"Optimal threshold = {opt_thresh:.2f}  (val F1={opt_f1:.4f})")
    print(f"  Accuracy : {opt_acc:.4f}  F1: {opt_f1:.4f}  MCC: {opt_mcc:.4f}")
    print(f"{'='*50}")
    print(f"Checkpoint: {best_ckpt}  |  Modulation log: ogm_epoch.csv")


if __name__ == "__main__":
    main()
