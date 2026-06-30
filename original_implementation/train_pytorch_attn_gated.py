"""
AudioFuse++ — Attention Pooling + Gated Fusion (Direction 1.3)

Two architectural changes over the baseline:
  1. Attention pooling: replaces Global Average Pooling in the ViT branch with a
     learned Linear(192,1) + softmax over patches, so the model focuses on the
     most diagnostically relevant spectrogram patches.
  2. Gated fusion: replaces fixed concatenation with a per-sample scalar gate
     g = sigmoid(Linear([f_spec; f_wave_proj] → 1)), fusing as
     fused = g * f_spec + (1-g) * f_wave_proj. f_wave is projected 64→192 first
     so both branches live in the same space. Gate values are logged on the val
     set to analyse branch dominance.

Usage:
    python train_pytorch_attn_gated.py --train_csv data/train.csv \
        --val_csv data/val.csv --seeds 1 --output_dir outputs/pytorch_attn_gated/
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from train_pytorch import (
    Config, PCGDataset, PatchEmbed, TransformerBlock, WaveformCNN,
    sweep_threshold, DEVICE
)

C = Config()


# ── Modified ViT with attention pooling ─────────────────────────────────────
class SpectrogramViTAttn(nn.Module):
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
        self.attn_pool = nn.Linear(C.PROJ_DIM, 1)  # replaces GAP

    def forward(self, x):
        x = self.patch_embed(x)                          # (B, N, 192)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.norm(x)                                 # (B, N, 192)
        attn_w = torch.softmax(self.attn_pool(x), dim=1) # (B, N, 1)
        return (x * attn_w).sum(dim=1)                   # (B, 192)


# ── AudioFuse++ with gated fusion ────────────────────────────────────────────
class AudioFusePP(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_branch = SpectrogramViTAttn()
        self.wave_branch = WaveformCNN()
        self.wave_proj = nn.Linear(64, C.PROJ_DIM)       # 64 → 192
        self.gate = nn.Linear(C.PROJ_DIM * 2, 1)         # [f_spec; f_wave_proj] → scalar
        nn.init.zeros_(self.gate.bias)                    # start at g=0.5, avoid early saturation
        self.head = nn.Sequential(
            nn.Linear(C.PROJ_DIM, C.PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(C.PROJ_DIM, 1),
        )

    def forward(self, spec, wave):
        f_spec = self.spec_branch(spec)                           # (B, 192)
        f_wave = F.relu(self.wave_proj(self.wave_branch(wave)))   # (B, 192)
        g = torch.sigmoid(self.gate(torch.cat([f_spec, f_wave], dim=1)))  # (B, 1)
        fused = g * f_spec + (1 - g) * f_wave                    # (B, 192)
        return self.head(fused).squeeze(1), g.squeeze(1)          # logits, gate


# ── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, pos_weight, train=True, lambda_entropy=0.1):
    model.train(train)
    total_loss, all_probs, all_labels, all_gates = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for spec, wave, labels in tqdm(loader, leave=False):
            spec = spec.to(DEVICE)
            wave = wave.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, gates = model(spec, wave)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            if train:
                if lambda_entropy > 0.0:
                    # Entropy regularization: penalise gate being near 0 or 1.
                    # H(g) = -(g*log(g) + (1-g)*log(1-g)), maximised at g=0.5.
                    eps = 1e-6
                    gate_entropy = -(gates * (gates + eps).log() +
                                     (1 - gates) * (1 - gates + eps).log()).mean()
                    loss = loss - lambda_entropy * gate_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_gates.extend(gates.detach().cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc, np.array(all_gates)


def train_one_seed(args, seed: int, lambda_entropy: float = 0.1):
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
    print(f"pos_weight = {pos_weight.item():.4f}")

    model = AudioFusePP().to(DEVICE)
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
        tr_loss, tr_acc, tr_auc, _ = run_epoch(model, train_loader, optimizer, pos_weight, train=True, lambda_entropy=lambda_entropy)
        vl_loss, vl_acc, vl_auc, vl_gates = run_epoch(model, val_loader, optimizer, pos_weight, train=False, lambda_entropy=lambda_entropy)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch:3d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} auc={vl_auc:.4f} | "
              f"gate_mean={vl_gates.mean():.3f}")

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
    all_probs, all_labels, all_gates = [], [], []
    val_labels_arr = []
    with torch.no_grad():
        for spec, wave, labels in val_loader:
            logits, gates = model(spec.to(DEVICE), wave.to(DEVICE))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_gates.extend(gates.cpu().numpy())
            val_labels_arr.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    gates_arr = np.array(all_gates)

    # Save predictions + gate values
    preds_path = os.path.join(args.output_dir, f"val_preds_seed{seed}.csv")
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "gate": gates_arr}).to_csv(preds_path, index=False)

    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    opt_thresh, opt_f1 = sweep_threshold(y_true, y_prob)
    y_pred_opt = (y_prob > opt_thresh).astype(int)
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_mcc = matthews_corrcoef(y_true, y_pred_opt)

    # Gate analysis — which branch dominates per class
    gate_normal = gates_arr[y_true == 0].mean()    # g≈1 → spec dominates
    gate_abnormal = gates_arr[y_true == 1].mean()  # g≈0 → wave dominates

    print(f"\n{'='*55}")
    print(f"Seed {seed} Results [ATTN POOL + GATED FUSION] (threshold=0.50):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"Optimal threshold = {opt_thresh:.2f}  (val F1={opt_f1:.4f})")
    print(f"  Accuracy : {opt_acc:.4f}")
    print(f"  F1-Score : {opt_f1:.4f}")
    print(f"  MCC      : {opt_mcc:.4f}")
    print(f"Gate analysis (g≈1 → spec branch dominates):")
    print(f"  Normal   class mean gate = {gate_normal:.3f}")
    print(f"  Abnormal class mean gate = {gate_abnormal:.3f}")
    print(f"{'='*55}")

    return {
        "seed": seed,
        "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc,
        "opt_threshold": opt_thresh,
        "opt_accuracy": opt_acc, "opt_f1": opt_f1, "opt_mcc": opt_mcc,
        "gate_normal": gate_normal, "gate_abnormal": gate_abnormal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_attn_gated/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--lambda_entropy", type=float, default=0.1,
                        help="Entropy regularization weight for gate (0.0 = disabled)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    print(f"lambda_entropy = {args.lambda_entropy}")
    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# Training with seed={seed} [ATTN POOL + GATED FUSION]\n{'#'*60}")
        all_results.append(train_one_seed(args, seed, lambda_entropy=args.lambda_entropy))

    results_df = pd.DataFrame(all_results)
    if len(all_results) > 1:
        print("\nFinal Summary (mean ± std over seeds):")
        for col in ["accuracy", "f1", "roc_auc", "mcc"]:
            print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    main()
