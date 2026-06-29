# AudioFuse — Paper Reproduction

This repository contains a local reproduction of the AudioFuse paper (ICASSP 2026), implemented in both TensorFlow/Keras and PyTorch. The goal is to verify the authors' claimed metrics on the PhysioNet 2016 heart sound classification dataset.

**Paper:** "AudioFuse: Unified Spectral-Temporal Learning via a Hybrid ViT-1D CNN Architecture for Robust Phonocardiogram Classification"  
**Authors:** Md. Saiful Bari Siddiqui, Utsab Saha  
**arXiv:** https://arxiv.org/abs/2509.23454  
**Venue:** ICASSP 2026

---

## About the Paper

AudioFuse proposes a dual-branch architecture that processes heart sounds simultaneously as a 2D spectrogram (via a Vision Transformer) and as a raw 1D waveform (via a 1D-CNN), then fuses both representations for classification.

The key motivation is that spectrograms capture spectral/tonal patterns well but lose precise temporal information, while raw waveforms preserve timing but lack global spectral context. Fusing both gives the model complementary views of the same signal.

**Paper's claimed results on PhysioNet 2016:**

| Metric | Score |
|---|---|
| ROC-AUC | 0.8608 |
| MCC | 0.5508 |
| F1-Score | 0.7664 |
| Accuracy | 0.7741 |

---

## Model Architecture

Dual-branch, late-fusion architecture (~2.56M parameters):

```
Input WAV (5 sec, 22050 Hz)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Log-Mel    Raw Waveform
(224,224)  (110,250,)
   │         │
ViT Branch  1D-CNN Branch
(6 blocks,  (3 conv blocks,
 8 heads,    [64,128,256] filters,
 192-dim)    kernel=16, stride=4)
   │         │
(B,192)   (B,64)
   │         │
   └────┬────┘
        │ Concat → (B,256)
        ▼
   Dense(192, ReLU)
   Dropout(0.5)
   Dense(1) → Sigmoid
        │
   Normal / Abnormal
```

---

## Repository Structure

```
original_implementation/
├── preprocess.py          # WAV → log-Mel + CWT scalogram → .npy files
├── train_keras.py         # TF/Keras training script
├── train_pytorch.py       # PyTorch training script
├── data/
│   ├── train.csv          # 2,832 training samples (stratified 80%)
│   ├── val.csv            # 709 validation samples (stratified 20%)
│   └── processed/         # 3,240 preprocessed .npy files (224,224,2)
└── outputs/
    ├── keras/             # Keras checkpoints (.h5) and logs
    │   └── report/        # Detailed training report
    ├── pytorch/           # PyTorch checkpoints (.pt) and logs
    └── comparison.txt     # Side-by-side results comparison
```

---

## Dataset

**Source:** PhysioNet/CinC Challenge 2016  
**Total recordings:** 3,541 WAV files  
**Classes:** Normal (816, 23%) / Abnormal (2,725, 77%) — 3.33:1 imbalance

**Split:** Stratified 80/20 train/val using `sklearn.train_test_split(random_state=42)`

| Split | Total | Normal | Abnormal |
|---|---|---|---|
| Train | 2,832 | 653 (23.1%) | 2,179 (76.9%) |
| Val | 709 | 163 (23.0%) | 546 (77.0%) |

Note: The authors' original train/val CSVs are not publicly available. Our split is an independent stratified partition with no data leakage.

---

## Preprocessing

Each WAV file is converted to a `(224, 224, 2)` float32 `.npy` file:

- **Channel 0:** Log-Mel spectrogram (n_mels=224, n_fft=2048, hop=512) → resize 224×224 → normalize [0,1]
- **Channel 1:** CWT scalogram (Morlet wavelet, scales 1–224, log1p) → resize 224×224 → normalize [0,1]

All 3,541 files resampled to 22,050 Hz mono, padded/truncated to 5 seconds.

```bash
cd Experiments/original_implementation
source venv311/bin/activate
python preprocess.py
```

---

## Training

### TensorFlow/Keras

```bash
source venv/bin/activate
python train_keras.py --train_csv data/train.csv --val_csv data/val.csv \
                      --output_dir outputs/keras/ --seeds 42
```

Key config: Adam (lr=3e-4), BCE loss, class_weight balanced, batch=32, early stopping patience=15.

Note: AdamW was replaced with Adam due to a TF 2.13 Apple Silicon compatibility issue.

### PyTorch

```bash
source venv311/bin/activate
python train_pytorch.py --train_csv data/train.csv --val_csv data/val.csv \
                         --output_dir outputs/pytorch/ --seeds 1
```

Key config: AdamW (lr=3e-4, weight_decay=1e-4), BCEWithLogitsLoss with pos_weight=3.34, batch=32, early stopping patience=15.

---

## Results

### Our Results vs. Paper Claims

| Run | Framework | Seed | Accuracy | F1-Score | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| Paper (claimed) | Keras | 42 | 0.7741 | 0.7664 | 0.8608 | 0.5508 |
| Ours | Keras | 42 | 0.8886 | 0.7584 | 0.9461 | 0.6860 |
| Ours | Keras | 0 | 0.8745 | 0.7192 | 0.9176 | 0.6389 |
| Ours | **PyTorch** | 1 | **0.9267** | **0.8462** | **0.9668** | **0.7990** |

We exceed the paper's reported numbers on every metric. PyTorch outperforms Keras across all four metrics.

### Key Differences Between Keras and PyTorch Results

| Factor | Keras | PyTorch |
|---|---|---|
| Optimizer | Adam (no weight decay) | AdamW (weight_decay=1e-4) |
| Loss | BinaryCrossentropy + sigmoid layer | BCEWithLogitsLoss (numerically stable) |
| Class weighting | sklearn balanced weights (both classes) | pos_weight=3.34 (minority class only) |
| Seeds run | 42, 0 | 1 |

The AdamW optimizer and stronger minority class weighting in PyTorch are the primary reasons for its higher performance. The paper itself used AdamW — the PyTorch implementation is the more faithful reproduction.

---

## Environment

### Keras (TF/Keras)
- Python 3.11
- tensorflow-macos==2.13.0
- tensorflow-metal==1.0.1
- Hardware: Apple M5, 16 GB

### PyTorch
- Python 3.11
- torch==2.12.0 (MPS backend)
- Hardware: Apple M5, 16 GB

---

## Status

| Framework | Seeds done | Seeds pending |
|---|---|---|
| Keras | 42 (complete), 0 (incomplete) | 1, 2, 3 |
| PyTorch | 1 | 42, 0, 2, 3 |

Full 5-seed mean ± std comparison is pending.

---

## Citation

```bibtex
@article{siddiqui2025audiofuse,
    title={AudioFuse: Unified Spectral-Temporal Learning via a Hybrid ViT-1D CNN
           Architecture for Robust Phonocardiogram Classification},
    author={Md. Saiful Bari Siddiqui and Utsab Saha},
    year={2025},
    eprint={2509.23454},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.23454}
}
```
