"""
Build a clean, deduplicated, leakage-free train/val split (Direction 3.1, Step 2.5).

The original data/train.csv + data/val.csv pooled training-a..f PLUS the official
validation/ folder, which is entirely a SUBSET of training (301 recordings all
present in training-a..f). That caused ~12% train/val leakage (identical .npy in
both) plus duplicate rows within train.

This script rebuilds from the source of truth — the six training-a..f REFERENCE.csv
files (3240 unique recordings) — and writes a fresh stratified, recording-disjoint
split. It is NON-DESTRUCTIVE: it only writes new files (data/train_clean.csv,
data/val_clean.csv) and never touches the originals or any data folder.

REFERENCE.csv encodes label as -1 (normal) / 1 (abnormal); we map to 0 / 1 to match
the existing pipeline. Split is recording-disjoint by construction (we split the
unique-recording list), stratified by (subset, label). A fixed split_seed makes the
split itself reproducible; training seeds vary separately.

Usage:
    python make_clean_split.py --val_frac 0.2 --split_seed 42
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SUBSETS = ["a", "b", "c", "d", "e", "f"]


def load_master() -> pd.DataFrame:
    """One row per unique recording from training-a..f REFERENCE.csv files."""
    rows = []
    for s in SUBSETS:
        ref = os.path.join(DATA_DIR, f"training-{s}", "REFERENCE.csv")
        df = pd.read_csv(ref, header=None, names=["filename", "ref_label"])
        for _, r in df.iterrows():
            name = str(r["filename"]).strip()
            label = 0 if int(r["ref_label"]) == -1 else 1
            rows.append({
                "filename": name,
                "label": label,
                "filepath": f"{DATA_DIR}/training-{s}/{name}.wav",
                "npy_filepath": f"{DATA_DIR}/processed/{name}.npy",
                "subset": s,
            })
    master = pd.DataFrame(rows)
    # dedupe defensively (should already be unique across training subsets)
    master = master.drop_duplicates(subset="filename").reset_index(drop=True)
    return master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--out_train", default=f"{DATA_DIR}/train_clean.csv")
    ap.add_argument("--out_val", default=f"{DATA_DIR}/val_clean.csv")
    args = ap.parse_args()

    master = load_master()
    print(f"Unique recordings (training-a..f): {len(master)}")

    # verify every npy exists
    missing = [p for p in master["npy_filepath"] if not os.path.exists(p)]
    if missing:
        print(f"WARNING: {len(missing)} npy files missing, e.g. {missing[:3]}")
    else:
        print("All npy files present.")

    # stratify by (subset, label) so both class ratio and database mix are preserved
    strat = master["subset"] + "_" + master["label"].astype(str)
    train_df, val_df = train_test_split(
        master, test_size=args.val_frac, random_state=args.split_seed, stratify=strat
    )

    cols = ["filename", "label", "filepath", "npy_filepath"]
    train_df[cols].to_csv(args.out_train, index=False)
    val_df[cols].to_csv(args.out_val, index=False)

    # ---- verification report ----
    tr_names, vl_names = set(train_df["filename"]), set(val_df["filename"])
    tr_npy, vl_npy = set(train_df["npy_filepath"]), set(val_df["npy_filepath"])
    print(f"\n{'='*50}")
    print("CLEAN SPLIT VERIFICATION")
    print(f"  train: {len(train_df)}  val: {len(val_df)}  total: {len(train_df)+len(val_df)}")
    print(f"  filename overlap train<->val: {len(tr_names & vl_names)}  (must be 0)")
    print(f"  npy overlap train<->val:      {len(tr_npy & vl_npy)}  (must be 0)")
    def ratio(df):
        n = len(df); pos = int(df['label'].sum())
        return f"{pos}/{n} abnormal ({pos/n:.3f})"
    print(f"  train class balance: {ratio(train_df)}")
    print(f"  val   class balance: {ratio(val_df)}")
    print(f"  wrote: {args.out_train}, {args.out_val}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
