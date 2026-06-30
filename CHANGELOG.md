# AudioFuse Extension — Changelog

Format: newest entries first. Check off items as done. Note failed approaches.

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
