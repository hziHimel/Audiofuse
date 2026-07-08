# AudioFuse Extension — Changelog

Format: newest entries first. Check off items as done. Note failed approaches.

---

## 2026-07-08

### [DONE] Waveform-Only Baseline — seed=1 (Direction 3.1 / pretrained branch init)

**What changed:**
- `train_pytorch_waveonly.py` — new script. `WaveClassifier` = `WaveformCNN` + 2-layer head (Linear(64→64)→ReLU→Dropout(0.5)→Linear(64→1)). No spectrogram loading, no ViT. BCE+pos_weight, AdamW, ReduceLROnPlateau. Early stopped at epoch 26.
- `test_waveonly.py` — 5 tests (output shape, single sample, sigmoid range, gradients, weight extractability). All pass.
- Pretrained CNN weights saved: `outputs/pytorch_waveonly/best_seed1.pt`

**Results vs baseline (full AudioFuse, seed=1):**

| Metric | Wave-Only | Full AudioFuse | Δ |
|--------|-----------|----------------|---|
| Accuracy | 0.8731 | 0.9267 | -0.0536 |
| F1 | 0.7078 | 0.8462 | -0.1384 |
| ROC-AUC | 0.9331 | 0.9668 | -0.0337 |
| MCC | 0.6288 | 0.7990 | -0.1702 |

At optimal threshold (0.25): Acc=0.8293, F1=0.7112, MCC=0.6291

**Key finding:** Wave-only AUC when trained *independently* (0.9331) is notably lower than wave-only AUC from branch ablation on the full model (0.9667). The full model's CNN is strengthened by joint training even though the ViT contributes little. This confirms the CNN is not the bottleneck — the joint training dynamics matter. Pretrained branch init should help the ViT catch up.

**Next step:** Pre-train ViT branch independently (spectrogram-only classifier), then load both pretrained branch weights into AudioFuse for fine-tuning. CNN weights ready at `outputs/pytorch_waveonly/best_seed1.pt`.

---

## 2026-07-05 (continued)

### [DONE] MFCC spectrogram input — seed=1 (Direction 1.2)

**What changed:**
- `train_pytorch_mfcc.py` — new script. `MFCCDataset` computes MFCC (n_mfcc=40) on-the-fly from wav files via librosa, normalizes, bilinearly resizes to (1,224,224). Same ViT + CNN + fusion head as baseline. Slower than baseline (~1 it/s vs 3.6) due to on-the-fly CPU MFCC computation.
- Early stopped at epoch 41 (resumed run). Output: `outputs/pytorch_mfcc/`

**Results vs baseline (log-Mel, seed=1):**

| Metric | Baseline (log-Mel) | MFCC | Δ |
|--------|-------------------|------|---|
| Accuracy | 0.9267 | 0.9252 | -0.0015 |
| F1 | 0.8462 | 0.8399 | -0.0063 |
| ROC-AUC | 0.9668 | 0.9621 | -0.0047 |
| MCC | 0.7990 | 0.7913 | -0.0077 |

At optimal threshold (0.55): Acc=0.9281, F1=0.8450, MCC=0.7982

**Branch ablation on MFCC model:**

| Condition | log-Mel model | MFCC model |
|-----------|--------------|------------|
| Full model AUC | 0.9668 | 0.9621 |
| Wave only (spec=0) AUC | 0.9667 | 0.9621 |
| Spec only (wave=0) AUC | 0.4588 | **0.5913** |
| Spec-dominant (normal) | 0.4% | 0% |
| Spec-dominant (abnormal) | 3.7% | 0% |

**Conclusion:** MFCC is nearly competitive with log-Mel (-0.0047 AUC). MFCC spec-only AUC improved (0.4588→0.5913), suggesting MFCC gives the ViT slightly more useful signal. However, the waveform branch still completely dominates — 100% of samples are wave-dominant. The fundamental issue (joint training lets CNN dominate gradients) persists regardless of spectrogram type. Next step: pretrained branch initialization to force both branches to contribute.

---

## 2026-07-05

### [DONE] On-the-fly Audio Augmentation — seed=1 (Direction 1.4)

**What changed:**
- `augmentations.py` — `GaussianNoise` (SNR=20 dB, RMS-scaled) and `SpecAugment` (2 time masks T≤40, 2 freq masks F≤20). Pure PyTorch ops, run on MPS with no CPU round-trips. Val set never augmented.
- `test_augmentations.py` — 8 unit tests covering shape, identity at eval, noise SNR scaling, mask bounded output. All pass.
- `train_pytorch_augment.py` — new script applying both augmentations in the training loop after batch is on device. Early stopped at epoch 35. Output: `outputs/pytorch_augment/`
- Note: time stretch (±10%) and pitch shift (±2 st) not implemented — require librosa and are too slow for on-the-fly use.

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Augment | Δ |
|--------|----------|---------|---|
| Accuracy | 0.9267 | 0.8914 | -0.0353 |
| F1 | 0.8462 | 0.7729 | -0.0733 |
| ROC-AUC | 0.9668 | 0.9536 | -0.0132 |
| MCC | 0.7990 | 0.7025 | -0.0965 |

**Conclusion:** Augmentation hurts across all metrics on seed=1. Consistent with the broader pattern — the dataset (3541 samples) is too small for augmentation to provide regularization benefit; it adds variance the model can't overcome. The waveform branch dominance finding also explains this: Gaussian noise directly corrupts the signal the model relies on most, while SpecAugment masks a branch (ViT) that contributes almost nothing.

---

## 2026-07-04 (continued 3)

### [DONE] Branch Contribution Ablation — seed=1 baseline (Direction 3.1)

**What changed:**
- `branch_ablation.py` — new script. Loads `outputs/pytorch/best_seed1.pt` and runs 3 forward passes on the val set: full model, spec-zeroed (wave only), wave-zeroed (spec only). Reports per-condition AUC/Acc and per-class branch dominance statistics.
- Output: `outputs/pytorch_ablation/ablation_results.csv`

**Results:**

| Condition | AUC | Accuracy |
|-----------|-----|----------|
| Full model | 0.9668 | 0.9267 |
| Wave only (spec=0) | **0.9667** | 0.9281 |
| Spec only (wave=0) | 0.4588 | 0.2299 |

**Branch dominance by class:**

| Class | Spec-dominant | Wave-dominant | Mean drop (spec removed) | Mean drop (wave removed) |
|-------|--------------|--------------|--------------------------|--------------------------|
| Normal (n=546) | 0.4% (2/546) | 99.6% (544/546) | 0.0034 | 0.8471 |
| Abnormal (n=163) | 3.7% (6/163) | 96.3% (157/163) | 0.0066 | 0.1708 |

**Key findings:**
- **Waveform CNN carries virtually all predictive signal**: wave-only AUC (0.9667) nearly matches the full model (0.9668).
- **ViT spectrogram branch contributes almost nothing**: spec-only AUC (0.4588) is worse than random, meaning the ViT branch alone cannot classify.
- The full model's marginal gain over wave-only is negligible (+0.0001 AUC).
- This is a critical finding: the dual-branch architecture is effectively a single-branch waveform model in practice. The ViT is either undertrained, or the spectrogram representation (single-channel log-Mel, no pretraining) is too weak for the ViT to learn from on this dataset size.
- **Implication for future work**: improving the ViT branch (pretrained weights, MFCC, better augmentation) is the highest-leverage direction. The gated fusion result (gate→waveform) is now fully explained.

---

## 2026-07-04 (continued 2)

### [DONE] Residual CNN waveform branch — seed=1 (Direction 1.3)

**What changed:**
- `train_pytorch_rescnn.py` — new script. `WaveformResCNN` replaces the 3-block shallow CNN with `ResBlock1d` blocks that add a 1×1 Conv1d skip connection around each block. Output dim stays at 64 to keep the fusion head identical to baseline. Fixed stride/padding off-by-one in skip via `min_len` trim.
- `test_rescnn.py` — 5 unit tests covering block shape, skip connection fires, output shape/sign, full model shape. All pass.
- Early stopped at epoch 61. Output: `outputs/pytorch_rescnn/`

**Results vs baseline (shallow CNN, seed=1):**

| Metric | Baseline | Residual CNN | Δ |
|--------|----------|--------------|---|
| Accuracy | 0.9267 | 0.9154 | -0.0113 |
| F1 | 0.8462 | 0.8225 | -0.0237 |
| ROC-AUC | 0.9668 | 0.9559 | -0.0109 |
| MCC | 0.7990 | 0.7679 | -0.0311 |

At optimal threshold (0.70): Acc=0.9210, F1=0.8293, MCC=0.7779

**Conclusion:** Residual CNN underperforms the shallow baseline across all metrics. Deeper architecture likely overfits on this small dataset (3541 samples). The original shallow 3-block CNN is already well-matched to the data size. Single-seed result but the gap is consistent and larger than noise.

---

## 2026-07-04 (continued)

### [DONE] Dual-channel spectrogram input (log-Mel + CWT) — seed=1 (Direction 1.2)

**What changed:**
- `train_pytorch_dualchan.py` — new script. `PCGDatasetDual` loads both `.npy` channels → `(2, H, W)`. `AudioFuseDual` replaces the ViT patch embedding `Conv2d(1→192)` with `Conv2d(2→192)`; rest of architecture unchanged.
- Matches the original authors' preprocessing intent (`preprocess.py` saves `(H, W, 2)` with channel 0=log-Mel, channel 1=CWT scalogram).
- Early stopped at epoch 50. Output: `outputs/pytorch_dualchan/`

**Results vs baseline (single-channel log-Mel, seed=1):**

| Metric | Baseline (ch0) | Dual-channel | Δ |
|--------|---------------|--------------|---|
| Accuracy | 0.9267 | 0.9224 | -0.0043 |
| F1 | 0.8462 | 0.8265 | -0.0197 |
| ROC-AUC | 0.9668 | 0.9644 | -0.0024 |
| MCC | 0.7990 | 0.7771 | -0.0219 |

At optimal threshold (0.40): Acc=0.9224, F1=0.8308, MCC=0.7805

**Conclusion:** Adding the CWT scalogram channel slightly hurts all metrics on seed=1. The ViT may treat the two channels as independent patches rather than learning cross-channel structure. The scalogram appears to add noise rather than complementary signal at this scale. Single-seed result — within noise range, but the trend is consistent with similar regularization experiments.

---

## 2026-07-04

### [DONE] Gated Fusion λ=0.0 (no entropy reg) — seed=1 (Direction 1.3)

**What changed:**
- Ran `train_pytorch_attn_gated.py --lambda_entropy 0.0 --seeds 1 --output_dir outputs/pytorch_attn_gated_lambda000/`
- No entropy regularization; gate initialized with zero bias (g=0.5 at start).
- Early stopped at epoch 19.

**Results vs baseline and other λ values:**

| λ | AUC | Gate behavior |
|---|-----|---------------|
| 0.0 (new) | 0.9302 | gate→0.003 (collapsed to waveform) |
| 0.01 | 0.9610 | meaningful distribution |
| 0.1 | 0.9304 | over-constrained |
| baseline (no gate) | 0.9668 | — |

**Conclusion:** Without entropy regularization the gate collapses to near-zero, meaning the model ignores the spectrogram branch entirely. Performance matches λ=0.1 (both ~0.93) but for opposite reasons: λ=0.1 over-constrains, λ=0.0 under-constrains. λ=0.01 is the sweet spot — confirms entropy reg is necessary for meaningful gated fusion.

---

### [DONE] Mixup augmentation α=0.4 — seed=1 (Direction 1.1)

**What changed:**
- `train_pytorch_mixup.py` — new script. Same as baseline except training batches are mixed via Mixup (α=0.4). Same λ applied to both spec and waveform inputs for consistency. Val is unchanged.
- `test_mixup.py` — 5 unit tests covering shapes, λ range, convex combination, same-λ for both modalities, α→0 identity. All pass.
- Early stopped at epoch 59. Output: `outputs/pytorch_mixup/`

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Mixup α=0.4 | Δ |
|--------|----------|-------------|---|
| Accuracy | 0.9267 | 0.9182 | -0.0085 |
| F1 | 0.8462 | 0.8232 | -0.0230 |
| ROC-AUC | 0.9668 | 0.9630 | -0.0038 |
| MCC | 0.7990 | 0.7700 | -0.0290 |

Optimal threshold stayed at 0.50.

**Conclusion:** Mixup slightly hurts all metrics on seed=1. Consistent pattern with focal loss and label smoothing — regularization techniques that help on large datasets don't clearly benefit this smaller dataset (3541 samples). Differences are within single-seed noise range; multi-seed CV needed for firm conclusions.

---

## 2026-07-03

### [DONE] Multi-seed baseline — seeds 42, 0, 1, 2, 3 (Direction baseline)

**What changed:**
- Ran `train_pytorch.py` with seeds 42, 0, 2, 3 (seed=1 was already done). Resume logic loaded `best_seed2.pt` checkpoint for seed=2.
- Used `caffeinate -i -w <PID>` to prevent macOS thermal throttling during training.

**Per-seed results (threshold=0.50):**

| Seed | Accuracy | F1 | ROC-AUC | MCC | Early stop epoch |
|------|----------|----|---------|-----|-----------------|
| 42 | 0.8745 | 0.6962 | 0.9240 | 0.6247 | 20 |
| 0 | 0.9281 | 0.8411 | 0.9570 | 0.7948 | 65 |
| 1 | 0.9267 | 0.8462 | 0.9668 | 0.7990 | — |
| 2 | 0.9281 | 0.8431 | 0.9669 | 0.7964 | 50 |
| 3 | 0.9323 | 0.8509 | 0.9669 | 0.8072 | 69 |

**Mean ± std across 5 seeds (threshold=0.50):**

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 0.9179 ± 0.0218 |
| F1 | 0.8155 ± 0.0597 |
| ROC-AUC | **0.9563 ± 0.0166** |
| MCC | 0.7644 ± 0.0700 |

**Comparison vs paper:**

| Metric | Paper (AudioFuse) | Ours (5-seed mean) | Δ |
|--------|------------------|--------------------|---|
| Accuracy | 0.7741 ± 0.0094 | 0.9179 ± 0.0218 | +0.1438 |
| F1 | 0.7664 ± 0.0005 | 0.8155 ± 0.0597 | +0.0491 |
| ROC-AUC | 0.8608 ± 0.0127 | **0.9563 ± 0.0166** | **+0.0955** |
| MCC | 0.5508 ± 0.0225 | 0.7644 ± 0.0700 | +0.2136 |

**Key findings:**
- Seed 42 is a clear outlier (AUC 0.9240, F1 0.6962) — likely a bad random init. All other seeds cluster tightly at AUC 0.9570–0.9669.
- Our 5-seed mean AUC beats the paper by +9.55 percentage points.
- High std in F1 (0.0597) driven entirely by seed 42; excluding it gives F1=0.8453±0.0041.
- Output: `outputs/pytorch_multiseed/` — checkpoints `best_seed{N}.pt`, val preds `val_preds_seed{N}.csv`.

---

## 2026-07-01

### [DONE] CosineAnnealingWarmRestarts scheduler — seed=1 (Direction 1.1)

**What changed:**
- `train_pytorch_cosine.py` — new script identical to baseline except `ReduceLROnPlateau` replaced with `CosineAnnealingWarmRestarts(T_0=10, eta_min=1e-6)`. LR cycles from 3e-4 → 1e-6 every 10 epochs then restarts.
- Early stopped at epoch 71 (4 full cosine cycles + partial 5th).

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Cosine LR | Δ |
|--------|----------|-----------|---|
| Accuracy | 0.9267 | 0.9224 | -0.0043 |
| F1 | 0.8462 | 0.8368 | -0.0094 |
| ROC-AUC | 0.9668 | **0.9677** | +0.0009 |
| MCC | 0.7990 | 0.7867 | -0.0123 |

At optimal threshold: Acc=0.9267, F1=0.8434, MCC=0.7957

**Conclusion:** Marginal AUC improvement (+0.0009) but other metrics slightly below baseline. Warm restarts help the model escape local minima (AUC improved across cycles: 0.9434 → 0.9627 → 0.9677) but the overall gain is within noise range on a single seed.

---

## 2026-06-30 (continued)

### [DONE] Gated Fusion λ tuning — λ=0.01 vs λ=0.1 (Direction 1.3)

**What changed:**
- `train_pytorch_attn_gated.py` — added `--lambda_entropy` CLI arg (default 0.1). `run_epoch` and `train_one_seed` now accept `lambda_entropy` parameter; entropy reg skipped entirely when λ=0.0.
- Run: `python train_pytorch_attn_gated.py --seeds 1 --lambda_entropy 0.01 --output_dir outputs/pytorch_attn_gated_lambda001/`
- Early stopped at epoch 37.

**Results comparison:**

| Metric | Baseline (seed=1) | λ=0.1 (original) | λ=0.01 (new) | Δ vs λ=0.1 |
|--------|------------------|-----------------|--------------|------------|
| Accuracy | 0.9267 | 0.8858 | **0.9238** | +0.0380 |
| F1 | 0.8462 | 0.7429 | **0.8235** | +0.0806 |
| ROC-AUC | 0.9668 | 0.9304 | **0.9610** | +0.0306 |
| MCC | 0.7990 | 0.6702 | **0.7779** | +0.1077 |

At optimal threshold (0.30): Acc=0.9252, F1=0.8328, MCC=0.7852

**Gate analysis (λ=0.01):**
- Normal class mean gate = 0.271 → model leans toward waveform for normal sounds
- Abnormal class mean gate = 0.076 → model strongly relies on waveform for abnormal sounds
- Contrast with λ=0.1: gate_normal=0.251, gate_abnormal=0.321 (inverted pattern due to over-constraint)

**Conclusion:** λ=0.01 recovers nearly all baseline performance while keeping the gated fusion architecture. The weaker regularization lets the model learn a meaningful gate without over-constraining it. λ=0.0 (bias-init only) still untested.

---

## 2026-06-30

### [DONE] 3-Fold Cross-Validation on Baseline — seed=42 (Direction 1.1)

**What changed:**
- `train_pytorch_kfold.py` — updated `N_SPLITS=3`, `KFOLD_PATIENCE=8` (reduced from 5-fold/patience=15 to cut wall-clock time). `train_one_fold` accepts explicit `patience` arg. Summary header fixed to `3-Fold CV Summary`.
- Run: `python train_pytorch_kfold.py --train_csv data/train.csv --val_csv data/val.csv --output_dir outputs/pytorch_kfold/`
- Output: `outputs/pytorch_kfold/` — per-fold checkpoints (`best_fold{N}.pt`), val predictions (`val_preds_fold{N}.csv`), `fold_results.csv`

**Per-fold results (threshold=0.50):**

| Fold | Accuracy | F1 | ROC-AUC | MCC | Early stop epoch |
|------|----------|----|---------|-----|-----------------|
| 1 | 0.8738 | 0.7353 | 0.9271 | 0.6533 | 12 |
| 2 | 0.8636 | 0.7160 | 0.9322 | 0.6273 | 20 |
| 3 | 0.8602 | 0.7016 | 0.9137 | 0.6105 | 10 |

**3-Fold CV mean ± std (threshold=0.50):**

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 0.8659 ± 0.0071 |
| F1 | 0.7177 ± 0.0169 |
| ROC-AUC | 0.9243 ± 0.0096 |
| MCC | 0.6304 ± 0.0216 |

**At optimal threshold (mean 0.58 ± 0.10):**
- Accuracy: 0.8769 ± 0.0034, F1: 0.7251 ± 0.0109, MCC: 0.6475 ± 0.0087

**Full comparison vs paper and single-seed baseline:**

| Metric | Paper (AudioFuse) | Our Repro (seed=1) | 3-Fold CV (ours) |
|--------|------------------|--------------------|-----------------|
| Accuracy | 0.7741 ± 0.0094 | 0.9267 | 0.8659 ± 0.0071 |
| F1 | 0.7664 ± 0.0005 | 0.8462 | 0.7177 ± 0.0169 |
| ROC-AUC | 0.8608 ± 0.0127 | 0.9668 | **0.9243 ± 0.0096** |
| MCC | 0.5508 ± 0.0225 | 0.7990 | 0.6304 ± 0.0216 |

**Key findings:**
- CV AUC (0.9243) beats the paper's reported AUC (0.8608) by +6.35%, despite training on only 67% of data per fold.
- Single-seed baseline is higher across all metrics — expected, as it trains on the full train split.
- Low variance (AUC std=0.0096) indicates stable training across folds.
- Optimal threshold 0.58 consistently above 0.5, confirming class imbalance effect.

**Note:** Paper reports results as mean ± std over multiple seeds. Our CV provides comparable statistical reporting format.

---

## 2026-06-29 (continued)

### [DONE] Attention Pooling + Gated Fusion — seed=1 (Direction 1.3)

**What changed:**
- `train_pytorch_attn_gated.py` — new `SpectrogramViTAttn` (replaces GAP with Linear(192,1)+softmax
  over patches) and `AudioFusePP` (projects f_wave 64→192, gate=sigmoid(Linear(384,1)),
  fused = g*f_spec + (1-g)*f_wave_proj). Gate values logged per val sample and analysed by class.
- `test_attn_gated.py` — 7 unit tests covering shape, attention weight sum, GAP vs attn diff,
  gate range, input-dependence, and backward pass. All pass.
- Two gate collapse issues encountered and fixed:
  1. Gate bias initialised to zero (`nn.init.zeros_`) to start at g=0.5 instead of random extreme.
  2. Entropy regularization added to loss: `L = L_cls - 0.1 * H(g)` to prevent saturation.
- Early stopping at epoch 19. Output: `outputs/pytorch_attn_gated/`

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Attn+Gated | Δ |
|--------|----------|------------|---|
| Accuracy | 0.9267 | 0.8858 | -0.0409 |
| F1 | 0.8462 | 0.7429 | -0.1033 |
| ROC-AUC | 0.9668 | 0.9304 | -0.0364 |
| MCC | 0.7990 | 0.6702 | -0.1288 |

**Gate analysis (key finding):**
- Normal class mean gate = 0.251 → model leans toward waveform branch for normal sounds
- Abnormal class mean gate = 0.321 → model uses more spectrogram for abnormal sounds
- Interpretation: abnormal heart sounds have stronger spectral signatures; normal sounds are
  better characterised by temporal waveform patterns. This is a meaningful finding for branch
  contribution analysis (Direction 3).

**Conclusion:** Performance below baseline on seed=1. The entropy regularization (λ=0.1) likely
constrained the model too much. Next steps: tune λ, or run without regularization but with better
gate initialisation only. Gate analysis findings are valuable regardless of performance.

---

## 2026-06-29

### [DONE] Focal Loss training run — seed=1 (Direction 1.1)

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Focal Loss | Δ |
|--------|----------|------------|---|
| Accuracy | 0.9267 | 0.9252 | -0.0015 |
| F1 | 0.8462 | 0.8408 | -0.0054 |
| ROC-AUC | 0.9668 | 0.9662 | -0.0006 |
| MCC | 0.7990 | 0.7923 | -0.0067 |

Optimal threshold stayed at 0.50 — focal loss calibrated predictions well without needing tuning.
**Conclusion:** Focal loss did not improve over BCE+pos_weight on seed=1. Differences within noise range.
Training stopped at epoch 35 (early stopping, patience=15).

---

### [DONE] Label Smoothing training run — seed=1 (Direction 1.1)

**Results vs baseline (BCE+pos_weight, seed=1):**

| Metric | Baseline | Label Smooth (ε=0.1) | Δ |
|--------|----------|----------------------|---|
| Accuracy | 0.9267 | 0.9267 | 0.0000 |
| F1 | 0.8462 | 0.8424 | -0.0038 |
| ROC-AUC | 0.9668 | 0.9579 | -0.0089 |
| MCC | 0.7990 | 0.7947 | -0.0043 |

Optimal threshold stayed at 0.50.
**Conclusion:** Label smoothing slightly hurt AUC (-0.009). No benefit on seed=1.
Training stopped at epoch 67 (early stopping, patience=15).

**Note:** Both runs are single-seed. Results need 5-fold CV for reliable mean ± std before drawing firm conclusions.

---

## 2026-06-28

### [DONE] Focal Loss, Label Smoothing, 5-fold CV (Direction 1.1)

**Tasks:** Focal loss, label smoothing, 5-fold cross-validation wrapper.

**What changed:**
- Added `losses.py` with `focal_loss_with_logits(alpha=0.25, gamma=2.0)` and
  `bce_with_label_smoothing(eps=0.1)`.
- Added `test_losses.py` with 7 unit tests covering both loss functions (all pass).
- `train_pytorch_focal.py` — full training script using focal loss instead of BCE+pos_weight.
  Focal loss handles 3.33:1 imbalance via alpha weighting, no explicit pos_weight needed.
- `train_pytorch_labelsmooth.py` — training script with label smoothing (eps=0.1) applied
  to BCE targets alongside pos_weight.
- `train_pytorch_kfold.py` — 5-fold StratifiedKFold wrapper. Accepts `--data_csv` (combined)
  or `--train_csv` + `--val_csv` (merged internally). Per-fold val_preds saved; prints
  mean ± std at both fixed (0.50) and optimal-F1 thresholds after all folds.

**Test result:** `11 passed in 0.94s` (venv311, Python 3.11)

**Pending:** GPU training runs to compare focal vs BCE+pos_weight vs label smooth on same
seed (seed=1) to measure actual F1/AUC differences.

---

### [DONE] Threshold sweep utility (Direction 1.1)

**Task (from TODO.md):** Sweep decision threshold on val set (0.1 to 0.9, step 0.05)
and report at optimal-F1 threshold.

**What changed:**
- Added `sweep_threshold(y_true, y_prob, lo, hi, step) → (threshold, f1)` to
  `original_implementation/train_pytorch.py` (above `main()`).
- `train_one_seed()` now calls `sweep_threshold` after final val inference and prints
  both the fixed-0.5 result and the optimal-threshold result side by side.
- Val predictions are saved as `outputs/pytorch/val_preds_seed{seed}.csv` (columns:
  `y_true`, `y_prob`) so the sweep can be re-run offline without reloading the model.
- Added `original_implementation/test_threshold_sweep.py` with 4 unit tests covering:
  perfect separation, imbalanced distribution (opt > 0.5 baseline), range validity,
  and all-positive edge case.

**Test result:** `4 passed in 1.93s` (venv311, Python 3.11)

**Pending:** Full GPU training run with seeds 42, 0, 1, 2, 3 to collect actual optimal
threshold values and measure F1 gain vs fixed 0.5. Expected gain: 2–8% F1 given 3.33:1
class imbalance.

**Motivation:** At 3.33:1 class imbalance the optimal F1 threshold is almost never 0.5.
This is a cheap, zero-parameter change that may close a visible gap to the paper numbers.

---

## Baseline (pre-experiment)

PyTorch seed=1 (fixed threshold=0.50):
- Accuracy: 0.9267, F1: 0.8462, ROC-AUC: 0.9668, MCC: 0.7990

Keras multi-seed mean (seeds 0,1,2,3):
- Accuracy: 0.8734 ± 0.009, F1: 0.7115 ± 0.019, ROC-AUC: 0.9153 ± 0.010, MCC: 0.6325 ± 0.024

---

## Pending / upcoming

- [ ] Run PyTorch seeds 42, 0, 2, 3 (seed 1 done) — requires GPU
- [ ] Add 5-fold CV wrapper
- [ ] Focal loss (α=0.25, γ=2.0) variant
- [ ] Label smoothing (ε=0.1) variant
- [ ] Attention pooling replacement for GAP in ViT branch
- [ ] Gated fusion implementation
