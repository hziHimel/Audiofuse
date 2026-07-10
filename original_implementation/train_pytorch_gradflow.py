"""
AudioFuse — Random-Init Training with Gradient-Flow Logging (Direction 3.1)

Trains the standard AudioFuse from random initialization (identical setup to the
baseline) but logs per-branch gradient L2 norms at every training step. Produces:
  - grad_flow.csv       : per-step grad_norm_spec, grad_norm_wave, ratio
  - grad_flow_epoch.csv : per-epoch mean grad norms + val AUC
  - grad_flow.png       : plot of per-branch gradient norms over training

Purpose: empirically prove that the waveform CNN dominates the gradient signal
early in joint training while the ViT branch receives vanishing gradients — the
root cause of the near-random spec-only ablation AUC (0.4588).

Usage:
    python train_pytorch_gradflow.py --train_csv data/train.csv \
        --val_csv data/val.csv --seed 1 --output_dir outputs/pytorch_gradflow/
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from train_pytorch import Config, AudioFuse, PCGDataset, DEVICE
from grad_flow import branch_grad_norms

C = Config()


def run_train_epoch(model, loader, optimizer, pos_weight, step_records, epoch):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for spec, wave, labels in tqdm(loader, leave=False):
        spec, wave, labels = spec.to(DEVICE), wave.to(DEVICE), labels.to(DEVICE)
        logits = model(spec, wave)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()

        # log per-branch gradient norms BEFORE the optimizer step
        norms = branch_grad_norms(model.spec_branch, model.wave_branch)
        norms["epoch"] = epoch
        step_records.append(norms)

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


def plot_grad_flow(epoch_df, out_path):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epoch_df["epoch"], epoch_df["grad_norm_wave"],
             label="Wave CNN grad norm", color="#d62728", linewidth=2)
    ax1.plot(epoch_df["epoch"], epoch_df["grad_norm_spec"],
             label="Spec ViT grad norm", color="#1f77b4", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean per-step gradient L2 norm")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.set_title("Per-branch gradient flow during random-init joint training")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  default="data/train.csv")
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_gradflow/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
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
    print(f"pos_weight={pos_weight.item():.4f}  [random-init + gradient-flow logging]")

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    step_records = []
    epoch_records = []
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_ckpt = os.path.join(args.output_dir, f"best_seed{args.seed}.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = run_train_epoch(model, train_loader, optimizer,
                                                   pos_weight, step_records, epoch)
        vl_loss, vl_acc, vl_auc = run_val_epoch(model, val_loader, pos_weight)
        scheduler.step(vl_loss)

        # per-epoch mean gradient norms
        ep_steps = [r for r in step_records if r["epoch"] == epoch]
        mean_spec = np.mean([r["grad_norm_spec"] for r in ep_steps])
        mean_wave = np.mean([r["grad_norm_wave"] for r in ep_steps])
        finite_ratios = [r["grad_ratio_wave_over_spec"] for r in ep_steps
                         if np.isfinite(r["grad_ratio_wave_over_spec"])]
        mean_ratio = np.mean(finite_ratios) if finite_ratios else float("nan")
        epoch_records.append({
            "epoch": epoch,
            "grad_norm_spec": mean_spec, "grad_norm_wave": mean_wave,
            "grad_ratio_wave_over_spec": mean_ratio,
            "val_auc": vl_auc, "val_acc": vl_acc,
        })

        print(f"Epoch {epoch:3d} | train auc={tr_auc:.4f} | val auc={vl_auc:.4f} | "
              f"grad_norm wave={mean_wave:.4f} spec={mean_spec:.4f} "
              f"ratio(w/s)={mean_ratio:.2f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= C.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # save logs
    pd.DataFrame(step_records).to_csv(
        os.path.join(args.output_dir, "grad_flow.csv"), index=False)
    epoch_df = pd.DataFrame(epoch_records)
    epoch_df.to_csv(os.path.join(args.output_dir, "grad_flow_epoch.csv"), index=False)

    plot_path = os.path.join(args.output_dir, "grad_flow.png")
    plot_grad_flow(epoch_df, plot_path)

    print(f"\n{'='*55}")
    print("Gradient-flow summary (mean per-step norms):")
    print(f"  Epoch 1  : wave={epoch_df.iloc[0]['grad_norm_wave']:.4f}  "
          f"spec={epoch_df.iloc[0]['grad_norm_spec']:.4f}  "
          f"ratio={epoch_df.iloc[0]['grad_ratio_wave_over_spec']:.2f}")
    print(f"  Final ep : wave={epoch_df.iloc[-1]['grad_norm_wave']:.4f}  "
          f"spec={epoch_df.iloc[-1]['grad_norm_spec']:.4f}  "
          f"ratio={epoch_df.iloc[-1]['grad_ratio_wave_over_spec']:.2f}")
    print(f"  Mean ratio (all epochs): {epoch_df['grad_ratio_wave_over_spec'].mean():.2f}")
    print(f"{'='*55}")
    print(f"Saved: grad_flow.csv, grad_flow_epoch.csv, grad_flow.png")


if __name__ == "__main__":
    main()
