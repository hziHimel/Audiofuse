"""
AudioFuse Preprocessing Script
Converts PhysioNet 2016 WAV files to log-Mel spectrogram .npy files
and generates train/val metadata CSVs.

Usage:
    python preprocess.py --data_dir data/ --out_dir data/processed/
"""

import os
import glob
import argparse
import zipfile

import numpy as np
import pandas as pd
import librosa
import cv2
import pywt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
SIGNAL_LENGTH_SECONDS = 5
MAX_LEN = SAMPLE_RATE * SIGNAL_LENGTH_SECONDS   # 110 250 samples
N_MELS = 224
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = 224
VAL_SPLIT = 0.2
RANDOM_STATE = 42


# ── Audio → features ───────────────────────────────────────────────────────
def load_wav(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(wav) > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)), "constant")
    return wav


def get_spectrogram(wav: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def get_scalogram(wav: np.ndarray) -> np.ndarray:
    scales = np.arange(1, N_MELS + 1)
    coeffs, _ = pywt.cwt(wav, scales, "morl")
    return np.log1p(np.abs(coeffs))


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-6)


def process_wav(wav_path: str, out_dir: str) -> str:
    """Process a single WAV → saves (IMG_SIZE, IMG_SIZE, 2) .npy; returns save path."""
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    save_path = os.path.join(out_dir, stem + ".npy")
    if os.path.exists(save_path):
        return save_path  # already done

    wav = load_wav(wav_path)

    spec = get_spectrogram(wav)
    spec = normalize(cv2.resize(spec, (IMG_SIZE, IMG_SIZE)))

    scalo = get_scalogram(wav)
    scalo = normalize(cv2.resize(scalo, (IMG_SIZE, IMG_SIZE)))

    combined = np.stack([spec, scalo], axis=-1).astype(np.float32)  # (H, W, 2)

    np.save(save_path, combined)
    return save_path


# ── Dataset loading ─────────────────────────────────────────────────────────
def load_physionet_df(data_root: str) -> pd.DataFrame:
    """
    Walk training-* and validation folders, read REFERENCE.csv files,
    return a combined DataFrame with columns: filename, label, filepath.
    Label: 1 = normal, -1 = abnormal in the original → we map to 0/1 (0=abnormal).
    """
    ref_files = glob.glob(os.path.join(data_root, "**", "REFERENCE.csv"), recursive=True)
    if not ref_files:
        raise FileNotFoundError(f"No REFERENCE.csv files found under {data_root}")

    dfs = []
    for ref_path in sorted(ref_files):
        folder = os.path.dirname(ref_path)
        df = pd.read_csv(ref_path, header=None, names=["filename", "label"])
        df["filepath"] = df["filename"].apply(
            lambda fn: os.path.join(folder, f"{fn}.wav")
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    # Original labels: 1 = normal, -1 = abnormal → remap: 1 = normal, 0 = abnormal
    combined["label"] = combined["label"].apply(lambda x: 1 if x == 1 else 0)

    # Drop rows whose WAV file doesn't exist
    exists = combined["filepath"].apply(os.path.exists)
    missing = (~exists).sum()
    if missing:
        print(f"  Warning: {missing} WAV files missing, dropping.")
    combined = combined[exists].reset_index(drop=True)
    return combined


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Directory with unzipped PhysioNet data")
    parser.add_argument("--out_dir", default="data/processed/", help="Where to save .npy files")
    parser.add_argument("--meta_dir", default="data/", help="Where to save CSV metadata files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)

    # ── 1. Unzip if needed ──────────────────────────────────────────────────
    for zip_name in ["training.zip", "validation.zip"]:
        zip_path = os.path.join(args.data_dir, zip_name)
        if os.path.exists(zip_path):
            print(f"Unzipping {zip_name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(args.data_dir)
            print(f"  Done.")

    # ── 2. Build DataFrame ─────────────────────────────────────────────────
    print("\nScanning for audio files...")
    df = load_physionet_df(args.data_dir)
    print(f"  Found {len(df)} recordings  "
          f"(normal={df['label'].sum()}, abnormal={(df['label']==0).sum()})")

    # ── 3. Train / val split (stratified) ──────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    # ── 4. Preprocess all WAVs (parallel) ─────────────────────────────────
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    print("\nPreprocessing WAV files → .npy spectrograms...")
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    wav_paths = all_df["filepath"].tolist()
    out_dir = args.out_dir

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Using {n_workers} parallel workers for {len(wav_paths)} files...")

    npy_paths = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_wav, p, out_dir): p for p in wav_paths}
        for future in tqdm(as_completed(futures), total=len(futures)):
            wav_p = futures[future]
            npy_paths[wav_p] = future.result()

    train_df["npy_filepath"] = train_df["filepath"].map(npy_paths)
    val_df["npy_filepath"] = val_df["filepath"].map(npy_paths)

    # ── 5. Save metadata CSVs ──────────────────────────────────────────────
    train_csv = os.path.join(args.meta_dir, "train.csv")
    val_csv = os.path.join(args.meta_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"\nSaved:\n  {train_csv}\n  {val_csv}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
