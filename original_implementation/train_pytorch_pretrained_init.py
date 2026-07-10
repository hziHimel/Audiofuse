"""
AudioFuse — Pretrained Branch Initialization (Direction 1.3)

Loads independently pretrained ViT and CNN branch weights into AudioFuse,
then fine-tunes end-to-end. Designed to solve the gradient dominance problem
where joint training from random init leaves the ViT branch undertrained.

Training strategy (3 phases):
  Phase 1 (epochs 1–FREEZE_EPOCHS): freeze both branches, train fusion head only
           → head learns to combine pretrained representations
  Phase 2 (epochs FREEZE_EPOCHS+1 onwards): unfreeze all, fine-tune end-to-end
           → all weights adapt jointly from a good starting point

Other improvements carried over from speconly v2:
  - Checkpoint saved on best val AUC
  - Class-balanced batch sampler
  - Lower LR for branches vs head (differential LR)
  - Linear warmup during phase 2
  - Plateau scheduler maximizes AUC

Usage:
    python train_pytorch_pretrained_init.py \
        --train_csv data/train.csv --val_csv data/val.csv \
        --spec_ckpt outputs/pytorch_speconly/best_seed1.pt \
        --wave_ckpt outputs/pytorch_waveonly/best_seed1.pt \
        --seeds 1 --output_dir outputs/pytorch_pretrained_init/
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from train_pytorch import Config, AudioFuse, PCGDataset, sweep_threshold, DEVICE
from train_pytorch_speconly import SpecClassifier
from train_pytorch_waveonly import WaveClassifier

C = Config()

FREEZE_EPOCHS  = 10    # epochs to train head-only before unfreezing branches
BRANCH_LR      = 5e-5  # low LR for pretrained branch weights during fine-tuning
HEAD_LR        = 3e-4  # higher LR for fusion head
WARMUP_EPOCHS  = 5     # linear warmup epochs after unfreezing (phase 2)
PATIENCE       = 25    # longer patience — joint fine-tuning is slower to converge


def load_pretrained_branches(model: AudioFuse, spec_ckpt: str, wave_ckpt: str):
    """Load independently pretrained branch weights into AudioFuse."""
    # Load ViT weights from SpecClassifier checkpoint
    spec_state = torch.load(spec_ckpt, map_location=DEVICE)
    vit_weights = {k.replace("spec_branch.", ""): v
                   for k, v in spec_state.items() if k.startswith("spec_branch.")}
    model.spec_branch.load_state_dict(vit_weights)
    print(f"Loaded ViT weights from: {spec_ckpt}")

    # Load CNN weights from WaveClassifier checkpoint
    wave_state = torch.load(wave_ckpt, map_location=DEVICE)
    cnn_weights = {k.replace("wave_branch.", ""): v
                   for k, v in wave_state.items() if k.startswith("wave_branch.")}
    model.wave_branch.load_state_dict(cnn_weights)
    print(f"Loaded CNN weights from: {wave_ckpt}")


def set_branches_frozen(model: AudioFuse, frozen: bool):
    for p in model.spec_branch.parameters():
        p.requires_grad = not frozen
    for p in model.wave_branch.parameters():
        p.requires_grad = not frozen


def make_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["label"].values
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


def make_optimizer(model: AudioFuse, phase2: bool = False):
    """Differential LR: lower for pretrained branches, higher for head."""
    if not phase2:
        return AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=C.WEIGHT_DECAY)
    return AdamW([
        {"params": model.spec_branch.parameters(), "lr": BRANCH_LR},
        {"params": model.wave_branch.parameters(), "lr": BRANCH_LR},
        {"params": model.head.parameters(),        "lr": HEAD_LR},
    ], weight_decay=C.WEIGHT_DECAY)


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
    val_df   = pd.read_csv(args.val_csv)

    sampler = make_balanced_sampler(train_df)
    train_loader = DataLoader(PCGDataset(train_df), batch_size=C.BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight={pos_weight.item():.4f}  freeze_epochs={FREEZE_EPOCHS}  "
          f"branch_lr={BRANCH_LR}  head_lr={HEAD_LR}  patience={PATIENCE}")

    model = AudioFuse().to(DEVICE)
    load_pretrained_branches(model, args.spec_ckpt, args.wave_ckpt)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_seed{seed}.pt")
    best_val_auc = 0.0
    epochs_no_improve = 0

    # Phase 1: freeze branches, train head only
    print(f"\n--- Phase 1: training head only (epochs 1–{FREEZE_EPOCHS}) ---")
    set_branches_frozen(model, frozen=True)
    optimizer = make_optimizer(model, phase2=False)
    scheduler = None

    for epoch in range(1, C.EPOCHS + 1):
        # Switch to phase 2 after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n--- Phase 2: fine-tuning all layers (epoch {epoch}+) ---")
            set_branches_frozen(model, frozen=False)
            optimizer = make_optimizer(model, phase2=True)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                          patience=10, min_lr=1e-6)

        # Linear warmup in phase 2
        if FREEZE_EPOCHS < epoch <= FREEZE_EPOCHS + WARMUP_EPOCHS:
            warmup_factor = (epoch - FREEZE_EPOCHS) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_factor

        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer,
                                             pos_weight, train=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, optimizer,
                                             pos_weight, train=False)

        if scheduler is not None and epoch > FREEZE_EPOCHS + WARMUP_EPOCHS:
            scheduler.step(vl_auc)

        phase = "P1-frozen" if epoch <= FREEZE_EPOCHS else "P2-finetune"
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} [{phase}] | lr={cur_lr:.2e} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} auc={vl_auc:.4f}")

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            if epoch > FREEZE_EPOCHS:  # only early-stop during phase 2
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}  (best val AUC={best_val_auc:.4f})")
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
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    opt_thresh, opt_f1 = sweep_threshold(y_true, y_prob)
    y_pred_opt = (y_prob > opt_thresh).astype(int)
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_mcc = matthews_corrcoef(y_true, y_pred_opt)

    print(f"\n{'='*50}")
    print(f"Seed {seed} Results [PRETRAINED-INIT] (threshold=0.50):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"Optimal threshold = {opt_thresh:.2f}  (val F1={opt_f1:.4f})")
    print(f"  Accuracy : {opt_acc:.4f}")
    print(f"  F1-Score : {opt_f1:.4f}")
    print(f"  MCC      : {opt_mcc:.4f}")
    print(f"{'='*50}")
    print(f"\nCheckpoint saved to: {best_ckpt}")

    return {
        "seed": seed,
        "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc,
        "opt_threshold": opt_thresh,
        "opt_accuracy": opt_acc, "opt_f1": opt_f1, "opt_mcc": opt_mcc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",  default="data/train.csv")
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--spec_ckpt",  default="outputs/pytorch_speconly/best_seed1.pt")
    parser.add_argument("--wave_ckpt",  default="outputs/pytorch_waveonly/best_seed1.pt")
    parser.add_argument("--output_dir", default="outputs/pytorch_pretrained_init/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# Training with seed={seed} [PRETRAINED-INIT]\n{'#'*60}")
        all_results.append(train_one_seed(args, seed))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    if len(all_results) > 1:
        print("\nFinal Summary (mean ± std over seeds):")
        for col in ["accuracy", "f1", "roc_auc", "mcc"]:
            print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")


if __name__ == "__main__":
    main()
