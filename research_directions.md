# Research Directions — AudioFuse Extension Ideas
**Base work:** AudioFuse reproduction on PhysioNet 2016 (ViT + 1D-CNN dual-branch)
**Date:** 2026-06-22

These are feasible experimental directions for original publication work, organized
across three broader research goals.

---

## Direction 1 — Improving Heart Sound Detection

Goal: push performance further or achieve competitive performance with fewer parameters.

### 1.1 Better Training Strategies

- **K-fold cross-validation (5 or 10 fold)**
  The paper used a single train/val split. K-fold would give more reliable mean ± std
  estimates and reduce the variance we are already seeing across seeds.
  Low implementation cost, high credibility gain.

- **Threshold optimization**
  We are using a fixed 0.5 decision threshold. With severe class imbalance (3.33:1),
  the optimal F1/MCC threshold is almost never 0.5. Sweeping the threshold on the
  val set and reporting at optimal-F1 threshold could close the gap significantly.

- **Focal Loss instead of BCE**
  Focal loss down-weights easy negatives and focuses training on hard misclassified
  examples. Well-suited for imbalanced datasets. Could replace the pos_weight approach.

- **Label smoothing**
  Softens hard 0/1 targets to reduce overconfidence. Cheap to implement, often
  improves calibration and generalization.

- **Mixup / CutMix augmentation**
  Interpolating between training pairs in both spectrogram and waveform space.
  Requires care for multimodal inputs — spec and wave must be mixed consistently.

- **Cosine annealing LR schedule with warm restarts**
  Replace ReduceLROnPlateau with cosine annealing (CosineAnnealingWarmRestarts).
  Often finds better minima by periodically resetting the learning rate.

### 1.2 Preprocessing Improvements

- **MFCC as an alternative to log-Mel**
  MFCCs apply the DCT on top of log-Mel, giving decorrelated features. Some papers
  report MFCC outperforms log-Mel specifically on PCG classification.

- **Gammatone filterbank**
  Perceptually motivated filterbank that mimics the human cochlea more closely than
  Mel. May capture heart sound harmonics differently.

- **STFT short-time energy / envelope features as a third channel**
  Add a (224,224,3) input where channel 2 is the short-time energy envelope. The
  envelope captures amplitude modulation patterns (S1/S2 timing) not well captured
  by spectrograms.

- **Better CWT parameters**
  Current scales are 1-224 uniform. Heart sounds are in 20-500 Hz range. Restricting
  scales to that band and using logarithmic spacing could improve the scalogram quality.

- **Segmentation-based preprocessing**
  Instead of fixed 5-second clips, segment recordings into individual cardiac cycles
  (using a peak detector or Springer's state machine). Train on per-cycle segments
  rather than arbitrary 5-second windows. This directly matches what the clinical
  task is about.

### 1.3 Architecture Improvements

- **Attention pooling instead of Global Average Pooling**
  GAP treats all 196 patches equally. A learnable attention weight vector can
  focus on the most diagnostically relevant patches. One extra Linear(192,1)
  layer with softmax over patches.

- **Cross-attention between branches (mid-fusion)**
  Instead of concatenating at the end, let the ViT tokens attend to the CNN
  feature map (and vice versa) mid-network. This allows branches to guide each
  other's representations before the fusion head. More expensive but potentially
  much more powerful.

- **Deeper waveform CNN with residual connections**
  Current 3-block CNN is very shallow. Adding skip connections (ResNet-style)
  between CNN blocks would allow gradients to flow better and allow more depth
  without degradation.

- **Lightweight / parameter-efficient variants**
  - Replace ViT with MobileViT (hybrid CNN-ViT, much fewer params)
  - Replace CNN branch with depthwise separable convolutions
  - Target: match AudioFuse accuracy at <1M parameters
  - Useful angle for deployment/edge device publications

### 1.4 Data-level Improvements

- **Audio augmentation pipeline**
  - Additive Gaussian noise (simulate recording conditions)
  - Time stretching ±10% (librosa.effects.time_stretch)
  - Pitch shifting ±2 semitones
  - SpecAugment on spectrogram (random frequency/time masking)
  - These are applied on-the-fly during training, not stored to disk

- **Semi-supervised learning**
  The PASCAL dataset (used for OOD evaluation in the paper) has unlabeled data.
  Using pseudo-labeling or consistency regularization could expand the effective
  training set.

---

## Direction 2 — Transfer Learning to Other Biomedical 1D Signals

Goal: investigate whether representations learned on heart sounds generalize to
other biosignals, analogous to ImageNet pretraining in vision.

The hypothesis: the waveform 1D-CNN branch learns low-level temporal patterns
(periodicity, transients, amplitude envelopes) that are signal-agnostic. Only
the higher-level representations are task-specific.

### 2.1 Target Domains

| Target Signal | Task | Dataset |
|---|---|---|
| ECG/EKG | Arrhythmia classification | PhysioNet MIT-BIH, PTB-XL |
| PPG | Atrial fibrillation detection | PhysioNet 2015, BIDMC |
| Lung sounds | Wheeze/crackle classification | ICBHI 2017 |
| EEG | Seizure detection | CHB-MIT Scalp EEG |
| EMG | Muscle activity classification | NinaPro |

Start with **ECG** (most data, most benchmarks) and **lung sounds** (closest
modality to heart sounds — same stethoscope domain).

### 2.2 Experimental Protocol

**Experiment A — Frozen feature extractor**
1. Take the AudioFuse model pretrained on PhysioNet heart sounds
2. Freeze all weights in both branches
3. Add a new classification head on top of the frozen fusion features
4. Train only the new head on the target signal dataset
5. Compare vs training from scratch on the target dataset

**Experiment B — Fine-tuning with frozen early layers**
1. Freeze only the first 1-2 CNN blocks and first 2 ViT blocks
2. Fine-tune the rest on the target dataset
3. Compare against full fine-tuning and frozen (Experiment A)

**Experiment C — Layer-wise transferability analysis**
1. Extract representations from each layer of the pretrained model
2. Train a simple linear probe (logistic regression) on each layer's output
3. Plot accuracy vs layer depth — shows which layers contain task-generic
   vs task-specific information
4. Inspired by "probing classifiers" used in NLP transfer learning literature

**Experiment D — Few-shot evaluation**
1. With frozen pretrained encoder, train the head on only 10/50/100 labeled
   examples from the target domain
2. Compare data efficiency vs a randomly initialized model trained on the same
   small labeled set
3. If pretraining helps with few examples, the transferability argument is strong

### 2.3 What Makes This Novel

Most transfer learning in biosignals focuses on:
- ECG → ECG (same modality)
- ImageNet pretrained ViT → biosignals (vision to biosignal)

Cross-modal biosignal transfer (heart sound → ECG, or heart sound → lung sound)
is relatively unexplored. The specific claim to test: **cardiac acoustic pretraining
captures temporal periodicity structures that are general across cyclic biosignals.**

### 2.4 Signals to Avoid (at first)

- fMRI, MRI (completely different modality, no temporal overlap)
- Text/NLP biosignals (no spectral-temporal structure)

---

## Direction 3 — Fusion Technique Improvements

Goal: understand and improve HOW the two branches are fused, and measure each
branch's contribution to the final prediction.

### 3.1 Contribution Analysis (Interpretability)

- **Ablation at inference time**
  For each val sample, run three forward passes: full model, spec-only (zero out
  wave), wave-only (zero out spec). Compare predictions to measure per-sample
  branch dominance. Aggregate across the dataset to get global contribution stats.

- **Gradient-based attribution per branch**
  Use Integrated Gradients or GradCAM to compute how much each branch's output
  contributed to the final prediction. Can be done per class (normal vs abnormal)
  to see if the model relies on different branches for different classes.

- **SHAP values on branch outputs**
  Treat the 192-dim ViT output and 64-dim CNN output as input features to the
  fusion head. Run SHAP on the fusion head to get per-feature importances, then
  aggregate by branch. Gives a scalar "branch importance score" per sample.

- **Attention weight visualization**
  Extract attention weights from each ViT transformer block. Visualize which
  spectrogram patches (time-frequency regions) the model attends to most for
  normal vs abnormal recordings. Check if abnormal attention concentrates on
  known murmur frequency bands (100-500 Hz).

### 3.2 Better Fusion Architectures

**Gated fusion (soft branch weighting)**
Instead of fixed concatenation, learn a scalar gate g ∈ [0,1] per sample:
```
g = sigmoid(Linear([f_spec; f_wave] → 1))
fused = g * f_spec + (1-g) * f_wave
```
The gate is input-dependent — for recordings where the waveform is cleaner,
the model can learn to rely more on it. Log g across the val set to analyze
which branch dominates per sample type.

**Cross-attention fusion**
Let the CNN features attend to ViT patch tokens (and vice versa):
```
f_spec_enhanced = CrossAttn(Q=f_wave, K=f_spec_patches, V=f_spec_patches)
f_wave_enhanced = CrossAttn(Q=f_spec, K=f_wave_seq, V=f_wave_seq)
fused = concat(f_spec_enhanced, f_wave_enhanced)
```
Allows the two branches to be informed by each other before fusion.
More parameters but potentially much stronger feature alignment.

**Bilinear pooling**
fused = f_spec ⊗ f_wave (outer product, flattened)
Captures pairwise interactions between every spec feature and every wave feature.
Computationally expensive (192×64=12,288 dim), but can be compressed with
compact bilinear pooling (count sketch trick).

**Mixture of Experts fusion**
Multiple fusion heads (e.g. 4 experts), each specializing in different patterns.
A gating network routes each sample to the most relevant expert.
Overkill for a binary task but potentially interesting for multi-class extensions.

### 3.3 Fusion Timing (Early vs Mid vs Late)

Current AudioFuse uses **late fusion** (branches run independently, features merged
at the very end). A systematic ablation:

| Fusion Type | Description |
|---|---|
| Early | Concatenate raw inputs before any processing |
| Mid-low | Fuse after CNN block 1 and ViT block 2 |
| Mid-high | Fuse after CNN block 3 and ViT block 5 |
| Late (current) | Fuse final feature vectors |
| Hierarchical | Fuse at multiple levels, combine all |

This produces a clean table showing how fusion timing affects performance and
which configuration is optimal — publishable as a standalone ablation study.

### 3.4 Contrastive Branch Training

Add a contrastive loss term between branches during training:
- Encourage the spec and wave representations of the SAME recording to be similar
- Encourage representations of DIFFERENT recordings to be dissimilar
- This forces both branches to learn a shared semantic space, potentially making
  the fusion more meaningful

```
L_total = L_classification + λ * L_contrastive(f_spec, f_wave)
```

Sweep λ ∈ {0.1, 0.5, 1.0} to find the right balance.

---

## Prioritization / Suggested Starting Points

| Idea | Impact | Effort | Novelty | Start here? |
|---|---|---|---|---|
| K-fold + threshold optimization | Medium | Low | Low | Yes — fixes evaluation |
| Segmentation-based preprocessing | High | Medium | Medium | Yes — clean contribution |
| Frozen PCG → lung sound transfer | High | Medium | High | Yes — strong novel angle |
| Gated fusion / contribution analysis | High | Medium | High | Yes — complements Direction 1 |
| Cross-attention fusion | High | High | Medium | Later |
| Focal loss + augmentation | Medium | Low | Low | Yes — easy wins |
| Layer-wise probing analysis | Medium | Medium | High | Yes — interpretability paper |
| Lightweight model (<1M params) | Medium | Medium | Medium | Later |

---

## Notes

- All three directions can share the same preprocessing pipeline and base model.
- Direction 2 (transfer learning) is the most novel and most likely to get traction
  in a top venue — biosignal transfer learning is an active and underexplored area.
- Direction 3 (fusion analysis) pairs well with Direction 1 — contribution analysis
  can be a section within a broader "improved AudioFuse" paper rather than standalone.
- Direction 1 improvements (segmentation, augmentation, gated fusion) together could
  form a single strong paper: "AudioFuse++" or similar.
