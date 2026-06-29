"""
AudioFuse — 5-Fold Cross-Validation wrapper (Direction 1.1)

Runs StratifiedKFold(n_splits=5) over a combined dataset to produce mean ± std
estimates for Accuracy, F1, ROC-AUC, and MCC. Each fold uses the standard
BCE + pos_weight loss from the base training script.

Usage:
    # Provide a single combined CSV (train + val concatenated):
    python train_pytorch_kfold.py --data_csv data/all.csv --output_dir outputs/pytorch_kfold/

    # Or provide separate CSVs (will be concatenated internally):
    python train_pytorch_kfold.py --train_csv data/train.csv --val_csv data/val.csv \
                                   --output_dir outputs/pytorch_kfold/
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from train_pytorch import Config, PCGDataset, AudioFuse, sweep_threshold, DEVICE

C = Config()
N_SPLITS = 5


def run_epoch(model, loader, optimizer, pos_weight, train=True):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []

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
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


def train_one_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   seed: int, output_dir: str) -> dict:
    torch.manual_seed(seed + fold_idx)  # different seed per fold for reproducibility
    np.random.seed(seed + fold_idx)

    train_loader = DataLoader(PCGDataset(train_df), batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

    model = AudioFuse().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_ckpt = os.path.join(output_dir, f"best_fold{fold_idx}.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, optimizer, pos_weight, train=True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader, optimizer, pos_weight, train=False)
        scheduler.step(vl_loss)

        print(f"  Fold {fold_idx} Epoch {epoch:3d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} auc={vl_auc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= C.PATIENCE:
                print(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
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

    preds_path = os.path.join(output_dir, f"val_preds_fold{fold_idx}.csv")
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

    print(f"\n  Fold {fold_idx} results (threshold=0.50): "
          f"acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} mcc={mcc:.4f}")
    print(f"  Fold {fold_idx} optimal threshold={opt_thresh:.2f}: "
          f"acc={opt_acc:.4f} f1={opt_f1:.4f} mcc={opt_mcc:.4f}")

    return {
        "fold": fold_idx,
        "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc,
        "opt_threshold": opt_thresh,
        "opt_accuracy": opt_acc, "opt_f1": opt_f1, "opt_mcc": opt_mcc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default=None,
                        help="Combined dataset CSV. If not provided, train_csv + val_csv are merged.")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/pytorch_kfold/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_csv:
        data_df = pd.read_csv(args.data_csv)
    else:
        data_df = pd.concat([pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)],
                            ignore_index=True)
    print(f"Total samples: {len(data_df)} | "
          f"pos: {(data_df['label']==1).sum()} | neg: {(data_df['label']==0).sum()}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    labels_array = data_df["label"].values

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_df, labels_array)):
        print(f"\n{'#'*60}\n# Fold {fold_idx + 1}/{args.n_splits}\n{'#'*60}")
        train_df = data_df.iloc[train_idx].reset_index(drop=True)
        val_df = data_df.iloc[val_idx].reset_index(drop=True)
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        result = train_one_fold(fold_idx + 1, train_df, val_df, args.seed, args.output_dir)
        fold_results.append(result)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(args.output_dir, "fold_results.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"5-Fold CV Summary (mean ± std, threshold=0.50):")
    for col in ["accuracy", "f1", "roc_auc", "mcc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print(f"\n5-Fold CV Summary (mean ± std, optimal threshold):")
    for col in ["opt_accuracy", "opt_f1", "opt_mcc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print(f"  opt_threshold : {results_df['opt_threshold'].mean():.2f} ± {results_df['opt_threshold'].std():.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
