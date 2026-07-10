# AudioFuse Extension — TODO

---

## Baseline (Pending from Reproduction)

- [ ] Run PyTorch ViT-only branch (seed 1) — requires GPU with batch size 256+
- [x] Run PyTorch AudioFuse seeds 42, 0, 2, 3 — mean AUC 0.9563 ± 0.0166 across 5 seeds (2026-07-03)
- [ ] Run Keras AudioFuse seeds 1, 2, 3 (seed 42 done, seed 0 incomplete)
- [ ] Rerun Keras AudioFuse seed 0 to completion
- [ ] Aggregate mean ± std across 5 seeds for Keras framework
- [ ] Update comparison.txt with full multi-seed results

---

## Direction 1 — Improving Heart Sound Detection

### 1.1 Better Training Strategies

- [x] Implement 5-fold cross-validation wrapper around the existing PyTorch training loop — `train_pytorch_kfold.py`; StratifiedKFold(5), reports mean ± std at fixed + optimal thresholds (2026-06-28)
- [x] After each fold, record per-fold Accuracy, F1, ROC-AUC, MCC and compute mean ± std — 3-fold CV complete; AUC 0.9243 ± 0.0096 (2026-06-30)
- [x] Sweep decision threshold on val set (0.1 to 0.9, step 0.05) and report at optimal-F1 threshold — `sweep_threshold()` added to train_pytorch.py; 4 tests pass (2026-06-28)
- [x] Replace BCE + pos_weight with Focal Loss (α=0.25, γ=2.0) and retrain AudioFuse — `train_pytorch_focal.py` + `losses.py`; 7 tests pass (2026-06-28)
- [x] Compare Focal Loss run vs pos_weight BCE run on same seed and fold — focal slightly worse on seed=1; within noise range (2026-06-29)
- [x] Add label smoothing (ε=0.1) to the BCE loss and measure effect on calibration — slightly hurt AUC (-0.009) on seed=1 (2026-06-29)
- [x] Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts (T_0=10) and retrain — marginal AUC gain (+0.0009); other metrics slightly below baseline (2026-07-01)
- [x] Implement Mixup augmentation (α=0.4) applied consistently to both spec and waveform inputs — `train_pytorch_mixup.py`; 5 tests pass (2026-07-04)
- [x] Measure Mixup effect on F1 and AUC vs no-augmentation baseline — AUC 0.9630 vs 0.9668 baseline (-0.0038); within noise on seed=1 (2026-07-04)

### 1.2 Preprocessing Improvements

- [x] Add MFCC (n_mfcc=40) as an alternative input to log-Mel spectrogram and retrain — `train_pytorch_mfcc.py` (2026-07-05)
- [x] Compare MFCC vs log-Mel ROC-AUC — MFCC AUC 0.9621 vs log-Mel 0.9668 (-0.0047); branch ablation shows ViT still 100% wave-dominant (2026-07-05)
- [ ] Implement Gammatone filterbank and generate spectrograms for all PhysioNet recordings
- [ ] Train ViT branch on Gammatone spectrograms and compare vs log-Mel baseline
- [x] Use both .npy channels (log-Mel + CWT scalogram) as 2-channel ViT input — `train_pytorch_dualchan.py`; AUC 0.9644 vs 0.9668 baseline (-0.0024); scalogram adds noise not signal (2026-07-04)
- [ ] Add short-time energy envelope as a 3rd channel to the spectrogram input (make it 224×224×3)
- [ ] Restrict CWT scales to 20–500 Hz range with logarithmic spacing and regenerate scalograms
- [ ] Retrain waveform branch on new CWT scalograms and compare AUC vs current
- [ ] Implement Springer's segmentation state machine (or a peak detector) to split recordings into cardiac cycles
- [ ] Generate per-cycle segments dataset from all PhysioNet 2016 recordings
- [ ] Train AudioFuse on per-cycle segments and compare vs fixed 5-second clip baseline

### 1.3 Architecture Improvements

- [x] Pre-train waveform CNN branch independently — `train_pytorch_waveonly.py`; wave-only AUC=0.9331 (vs 0.9667 from ablation on joint model); CNN weights saved at `outputs/pytorch_waveonly/best_seed1.pt` (2026-07-08)
- [x] Pre-train ViT branch independently (spectrogram-only classifier) — `train_pytorch_speconly.py` v2; AUC=0.9592 vs joint ablation 0.4588; smoking gun: ViT is capable alone but killed by joint training; weights saved at `outputs/pytorch_speconly/best_seed1.pt` (2026-07-09)
- [x] Initialize AudioFuse fusion model with pretrained branch weights, then fine-tune end-to-end — `train_pytorch_pretrained_init.py`; AUC=0.9677 vs 0.9668 baseline (+0.0009); F1/MCC beat baseline at optimal threshold (2026-07-11)
- [x] Run branch ablation on pretrained-init model — spec-only AUC 0.4588→0.9621; ViT now dominates 96.3% of abnormal decisions (was 3.7%); both branches genuinely complementary (2026-07-11)

- [x] Replace Global Average Pooling in ViT branch with attention pooling (Linear(192,1) + softmax over patches) — `SpectrogramViTAttn` in `train_pytorch_attn_gated.py` (2026-06-29)
- [x] Measure attention pooling effect on AUC vs GAP baseline (same seed) — AUC 0.9304 vs 0.9668 baseline; below baseline, likely due to entropy reg constraining model (2026-06-29)
- [x] Implement deeper waveform CNN with residual (skip) connections between the 3 blocks — `train_pytorch_rescnn.py`; 5 tests pass (2026-07-04)
- [x] Train residual CNN branch and compare vs original shallow CNN — AUC 0.9559 vs 0.9668 baseline (-0.0109); deeper arch overfits on small dataset (2026-07-04)
- [x] Implement gated fusion: g = sigmoid(Linear([f_spec; f_wave_proj] → 1)); fused = g*f_spec + (1-g)*f_wave_proj — `AudioFusePP` in `train_pytorch_attn_gated.py` (2026-06-29)
- [x] Log gate values g on the val set and analyze distribution (which branch dominates per class) — saved in val_preds_seed1.csv (2026-06-29)
- [x] Compare gated fusion AUC vs late concatenation fusion baseline — 0.9304 vs 0.9668; gate analysis shows abnormal sounds rely more on spectrogram (gate=0.321) vs normal (gate=0.251) (2026-06-29)
- [x] Tune entropy regularization λ (try 0.0, 0.01, 0.1) — λ=0.01 is optimal; λ=0.0 collapses gate, λ=0.1 over-constrains (2026-07-04)
- [ ] Implement cross-attention mid-fusion: CNN features attend to ViT patch tokens (and vice versa)
- [ ] Benchmark cross-attention fusion vs late fusion on AUC, F1, parameter count
- [ ] Prototype MobileViT replacement for the ViT branch and measure param count vs AUC tradeoff
- [ ] Prototype depthwise separable convolutions replacement for CNN branch
- [ ] Train lightweight variant targeting <1M total parameters and report accuracy vs param count

### 1.4 Data-level Improvements

- [x] Implement on-the-fly audio augmentation pipeline: GaussianNoise (SNR=20dB) + SpecAugment — `augmentations.py`; 8 tests pass (2026-07-05)
- [x] Apply augmentation consistently to both branches — `train_pytorch_augment.py`; augments applied after batch on device (2026-07-05)
- [x] Train AudioFuse with augmentation and compare vs no-augmentation — AUC 0.9536 vs 0.9668 baseline (-0.0132); hurts on small dataset (2026-07-05)
- [ ] Generate pseudo-labels for PASCAL unlabeled data using trained AudioFuse (confidence > 0.9)
- [ ] Retrain AudioFuse with PhysioNet + pseudo-labeled PASCAL and compare vs PhysioNet-only

---

## Direction 2 — Transfer Learning to Other Biomedical Signals

### 2.0 Spectrogram-domain Transfer (new idea — 2026-06-29)

- [ ] Freeze only the ViT (spectrogram) branch of the pretrained AudioFuse; attach a new head
- [ ] Convert target-domain signals (lung sounds, ECG) to log-Mel spectrograms in the same
      format as heart sound spectrograms (224×224, same mel bins and hop length)
- [ ] Train the new head on target-domain spectrograms — tests whether cardiac spectrogram
      representations transfer better than ImageNet ViT (closer domain match)
- [ ] Compare: frozen ViT spec branch vs frozen ImageNet ViT vs from-scratch ViT on target domain
- [ ] Repeat with frozen waveform CNN branch to isolate which branch transfers better
- [ ] Link to gate analysis: if abnormal gate≈spec, the spec branch should transfer richer features

### 2.1 Setup

- [ ] Download and preprocess PhysioNet MIT-BIH ECG dataset (resample, segment to fixed length)
- [ ] Download and preprocess ICBHI 2017 lung sound dataset (resample, segment to fixed length)
- [ ] Write a unified DataLoader that can load ECG or lung sound segments in the same format as heart sounds
- [ ] Verify that the waveform branch input shape is compatible with ECG and lung sound segment lengths

### 2.2 Experiment A — Frozen Feature Extractor

- [ ] Freeze all weights of pretrained AudioFuse (PhysioNet-trained)
- [ ] Attach a new classification head on top of frozen fusion features
- [ ] Train only the new head on MIT-BIH ECG (arrhythmia classification)
- [ ] Train only the new head on ICBHI lung sounds (wheeze/crackle classification)
- [ ] Record accuracy and AUC for each target domain and compare vs training from scratch

### 2.3 Experiment B — Fine-tuning with Partially Frozen Layers

- [ ] Freeze first 1–2 CNN blocks and first 2 ViT blocks; fine-tune rest on ECG
- [ ] Freeze first 1–2 CNN blocks and first 2 ViT blocks; fine-tune rest on lung sounds
- [ ] Compare partial fine-tune vs full fine-tune vs frozen (Experiment A) on both target domains

### 2.4 Experiment C — Layer-wise Transferability (Probing)

- [ ] Extract representations from each layer of pretrained AudioFuse (both branches)
- [ ] Train a logistic regression linear probe on each layer's output for the ECG task
- [ ] Train a logistic regression linear probe on each layer's output for the lung sound task
- [ ] Plot probe accuracy vs layer depth for each target domain (shows task-generic vs task-specific layers)

### 2.5 Experiment D — Few-shot Evaluation

- [ ] Sample 10, 50, 100 labeled examples from ECG dataset (stratified by class)
- [ ] Train frozen pretrained head on each sample size and record accuracy
- [ ] Train randomly initialized AudioFuse on each sample size and record accuracy
- [ ] Plot data efficiency curve: pretrained vs random init across {10, 50, 100} labeled examples

---

## Direction 3 — Fusion Technique Improvements & Analysis

### 3.1 Contribution Analysis

- [x] For each val sample, run 3 forward passes: full model, spec-zeroed, wave-zeroed — `branch_ablation.py` (2026-07-04)
- [x] Compute per-sample branch dominance (which branch drives the correct prediction) — wave-dominant: 99.6% normal, 96.3% abnormal (2026-07-04)
- [x] Aggregate branch dominance stats across val set by class — wave-only AUC=0.9667 ≈ full model; spec-only AUC=0.4588 (worse than random); ViT branch contributes almost nothing (2026-07-04)
- [ ] Implement Integrated Gradients on each branch output and compute per-branch attribution scores
- [ ] Visualize GradCAM attribution maps for normal vs abnormal samples from the ViT branch
- [ ] Run SHAP on the fusion head treating [f_spec (192-dim); f_wave (64-dim)] as input features
- [ ] Aggregate SHAP values by branch to produce a scalar branch importance score per sample
- [ ] Extract ViT attention weights from each transformer block and visualize attended patches
- [ ] Check whether abnormal-class attention concentrates on 100–500 Hz frequency bands in spectrogram

### 3.1.5 Branch Complementarity Analysis

- [ ] Error disagreement analysis: for each val sample classify into (wave✓/spec✓), (wave✓/spec✗), (wave✗/spec✓), (wave✗/spec✗) using wave-only and spec-only val predictions; report off-diagonal counts as evidence of complementarity
- [ ] CKA (Centered Kernel Alignment): collect f_spec (192-dim) and f_wave (64-dim) on val set from the full AudioFuse model; compute linear CKA score (0=dissimilar, 1=identical); low CKA = branches encode different information
- [ ] Repeat CKA before vs after pretrained-branch-init to see if joint fine-tuning collapses representations toward redundancy
- [ ] (Optional) Grad-CAM on ViT branch to visualize attended spectrogram patches; compare with CNN saliency on waveform to show qualitatively different focus regions

### 3.2 Fusion Timing Ablation

- [ ] Implement early fusion variant: concatenate raw spectrogram + waveform before any branch processing
- [ ] Implement mid-low fusion: fuse after CNN block 1 and ViT block 2, then continue processing jointly
- [ ] Implement mid-high fusion: fuse after CNN block 3 and ViT block 5
- [ ] Implement hierarchical fusion: fuse at multiple levels and combine all fusion outputs
- [ ] Train all fusion timing variants on the same 5-fold splits using the same hyperparameters
- [ ] Produce a comparison table: fusion timing vs Accuracy, F1, AUC, MCC, param count

### 3.3 Advanced Fusion Architectures

- [ ] Implement bilinear pooling fusion: f_spec ⊗ f_wave (outer product, flattened to 192×64=12,288 dim)
- [ ] Add compact bilinear pooling (count sketch trick) to compress bilinear output to 512 dim
- [ ] Train bilinear pooling fusion and compare vs concatenation baseline
- [ ] Implement Mixture of Experts fusion: 4 expert heads + gating network
- [ ] Train MoE fusion on AudioFuse and compare AUC vs single head
- [ ] Implement contrastive branch loss: L_total = L_cls + λ * L_contrastive(f_spec, f_wave)
- [ ] Sweep λ ∈ {0.1, 0.5, 1.0} and record AUC for each to find optimal contrastive weight

---

## Reporting & Publication Prep

- [ ] Write ablation table covering Direction 1.1–1.3 results (training strategies + architecture)
- [ ] Write transfer learning results table (Direction 2, Experiments A–D, both target domains)
- [ ] Write fusion timing + architecture comparison table (Direction 3.2–3.3)
- [ ] Write branch contribution analysis section with figures (Direction 3.1)
- [ ] Update comparison.txt with final best results across all experiments
