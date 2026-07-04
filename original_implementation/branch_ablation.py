"""
AudioFuse — Branch Contribution Ablation (Direction 3.1)

Loads the baseline checkpoint (best_seed1.pt) and runs 3 forward passes
on the val set:
  1. Full model  (spec + wave)
  2. Spec-zeroed (wave branch only — spec input replaced with zeros)
  3. Wave-zeroed (spec branch only — wave input replaced with zeros)

Reports per-class AUC, accuracy, and branch dominance (fraction of samples
where zeroing that branch causes the largest prediction drop).
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pytorch import PCGDataset, AudioFuse, Config, DEVICE

C = Config()


def run_forward(model, loader, zero_spec=False, zero_wave=False):
    """Single forward pass; optionally zero out one branch's input."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for spec, wave, labels in tqdm(loader, leave=False):
            spec, wave = spec.to(DEVICE), wave.to(DEVICE)
            if zero_spec:
                spec = torch.zeros_like(spec)
            if zero_wave:
                wave = torch.zeros_like(wave)
            logits = model(spec, wave)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--checkpoint", default="outputs/pytorch/best_seed1.pt")
    parser.add_argument("--output_dir", default="outputs/pytorch_ablation/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    loader = DataLoader(PCGDataset(val_df), batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = AudioFuse().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    print(f"Loaded checkpoint: {args.checkpoint}")

    print("\nRunning 3 forward passes...")
    probs_full, labels = run_forward(model, loader)
    probs_nospec,  _   = run_forward(model, loader, zero_spec=True)
    probs_nowave,  _   = run_forward(model, loader, zero_wave=True)

    # ── Per-condition metrics ────────────────────────────────────────────────
    for name, probs in [("Full model",  probs_full),
                         ("Wave only (spec=0)", probs_nospec),
                         ("Spec only (wave=0)", probs_nowave)]:
        acc = accuracy_score(labels, (probs > 0.5).astype(int))
        auc = roc_auc_score(labels, probs)
        print(f"\n{name}:")
        print(f"  AUC={auc:.4f}  Acc={acc:.4f}")

    # ── Per-sample branch dominance ──────────────────────────────────────────
    # Drop in prob when branch is zeroed; larger drop = that branch matters more
    drop_spec = probs_full - probs_nospec   # drop when spec is removed
    drop_wave = probs_full - probs_nowave   # drop when wave is removed

    # Sample is "spec-dominant" if removing spec hurts more than removing wave
    spec_dominant = (np.abs(drop_spec) > np.abs(drop_wave)).astype(int)

    results = pd.DataFrame({
        "y_true": labels,
        "prob_full": probs_full,
        "prob_nospec": probs_nospec,
        "prob_nowave": probs_nowave,
        "drop_spec": drop_spec,
        "drop_wave": drop_wave,
        "spec_dominant": spec_dominant,
    })
    results.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)

    # ── Aggregate by class ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("Branch dominance by class (fraction of samples where")
    print("removing that branch causes a larger prediction drop):")
    print("="*55)
    for cls, cls_name in [(0, "Normal"), (1, "Abnormal")]:
        mask = labels == cls
        n = mask.sum()
        spec_dom_frac = spec_dominant[mask].mean()
        wave_dom_frac = 1 - spec_dom_frac
        mean_drop_spec = np.abs(drop_spec[mask]).mean()
        mean_drop_wave = np.abs(drop_wave[mask]).mean()
        print(f"\n  {cls_name} (n={n}):")
        print(f"    Spec-dominant: {spec_dom_frac:.3f} ({spec_dom_frac*n:.0f}/{n} samples)")
        print(f"    Wave-dominant: {wave_dom_frac:.3f} ({wave_dom_frac*n:.0f}/{n} samples)")
        print(f"    Mean |drop| when spec removed: {mean_drop_spec:.4f}")
        print(f"    Mean |drop| when wave removed: {mean_drop_wave:.4f}")

    print(f"\nResults saved to {args.output_dir}/ablation_results.csv")


if __name__ == "__main__":
    main()
