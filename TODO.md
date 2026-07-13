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

#### Gradient Dominance — Comparative Study of Remedies (new — 2026-07-11)

Goal: turn the paper from "we found a fix" into "we systematically studied the problem and evaluated multiple remedies." Produce a comparison table: method vs full AUC, spec-only AUC, wave-only AUC, ViT-dominance %, training cost. Pretrained-init is one of several remedies.

- [x] **Gradient-flow instrumentation** — `grad_flow.py` + `train_pytorch_gradflow.py`; wave/spec grad-norm ratio climbs 3.03x (ep1) → 85.29x (final), mean 50.40x; ViT grad norm collapses 0.374→0.022; `grad_flow.png` centerpiece figure. 6 tests pass. Reusable for remedy comparison. (2026-07-11)
- [x] **OGM-GE (On-the-fly Gradient Modulation, Peng et al. CVPR 2022)** — `ogm.py` + `train_pytorch_ogm.py`; 9 tests pass. AUC=0.9674; spec-only ablation 0.4588→0.6155 (partial ViT activation); ViT-dominant abnormal 3.7%→33.1%. Partial fix — weaker than pretrained-init (0.9621 / 96.3%). Clean graded ordering established. (2026-07-12)
- [x] **Modality dropout** — `modality_dropout.py` + `train_pytorch_moddrop.py`; 7 tests pass. AUC=0.9667; spec-only ablation 0.4588→0.5933 but ViT-dominance DROPS to 0.6% abnormal (was 3.7%). Effectively a FAILED activation remedy — input-level dropout nudges ViT capability but not fusion utilization. Useful negative result. (2026-07-12)
- [ ] **PRIORITY — gradient flow DURING pretrained-init fine-tuning** (closes the mechanism loop): add `branch_grad_norm` logging (already in `grad_flow.py`) to `train_pytorch_pretrained_init.py` and rerun. We measured gradient flow on random-init (ratio 3x→85x, ViT dies) and on OGM-GE (pre/post modulation), but NOT on pretrained-init. Expected: with pretrained branches the wave/spec ratio stays balanced (~1–3x) instead of exploding to 85x → the SAME instrument shows the disease (random-init) and the cure (pretrained-init). Symmetric before/after figure directly tying mechanism (gradient dominance) to fix. Fold into the per-seed pretrained-init runs so we get the gradient trace per seed for free. ~15 min/run.
- [ ] **Decoupled/separate learning rates**: give ViT a much higher LR than CNN in joint training from scratch — tests whether the problem is purely convergence *speed*.
- [ ] **Gradient magnitude balancing**: normalize/rescale per-branch gradients to equal magnitude each step (cheaper than OGM-GE). Optional ablation point.
- [ ] Produce final comparison table across all remedies + random-init baseline; report which best activates the ViT (highest spec-only AUC) at what training cost.

#### Publication Sequencing Plan (2026-07-12) — do in this order

**Step 1 — finish the remedy comparison (in progress)**
- [x] Run branch ablation on the modality-dropout model — spec-only 0.5933, ViT-dom abnormal 0.6% (failed activation); slotted into 4-way table (2026-07-12)
- [ ] (optional, lower priority) decoupled-LR and gradient-magnitude-balancing remedies if time permits — otherwise the 3-remedy table (OGM-GE, modality-dropout, pretrained-init) vs baseline is already sufficient

**Step 2 — statistical rigor: 5-seed runs (do BEFORE writing; the main gap to submission)**

Decision: **5 seeds total = {1, 2, 3, 4, 5}**. Seed 1 already done. Fully rigorous
approach ("Option A"): pretrained-init gets its OWN branch pretraining per seed
(no reuse of seed-1 branch weights), so a reviewer cannot attribute the result to
one lucky initialization. Runs are sequential on the single MPS GPU (~2.5h/seed →
~10h for the 4 remaining seeds; run as an overnight batch). Use `caffeinate -i -w
<PID>` on every run.

*2.1 — Orchestration*
- [ ] Write `run_multiseed.py` (or a shell driver) that, for a given seed, runs the full pipeline in order and logs each to `logs/seed<N>_*.log`:
      1. baseline (`train_pytorch.py` or `train_pytorch_gradflow.py`) → checkpoint
      2. wave-only (`train_pytorch_waveonly.py`) → CNN weights for that seed
      3. spec-only (`train_pytorch_speconly.py`) → ViT weights for that seed
      4. pretrained-init (`train_pytorch_pretrained_init.py`) using THIS seed's branch weights
      5. OGM-GE (`train_pytorch_ogm.py`)
      6. modality-dropout (`train_pytorch_moddrop.py`)
      7. branch ablation (`branch_ablation.py`) on each of: baseline, pretrained-init, OGM-GE, modality-dropout
- [ ] Ensure every script writes seed-suffixed output dirs so seeds never overwrite each other

*2.2 — Execute*
- [ ] Run seeds 2, 3, 4, 5 through the full pipeline (seed 1 already complete)
- [ ] Sanity-check each seed's logs for early-stopping anomalies / NaNs before trusting numbers

*2.3 — Aggregate*
- [ ] Collect per-seed metrics into one CSV: for each remedy × seed record full AUC, spec-only ablation AUC, wave-only ablation AUC, ViT-dominance% (normal/abnormal), F1, MCC
- [ ] Compute mean ± std across the 5 seeds for every cell of the 4-way remedy comparison table
- [ ] Confirm the branch-specialization result (ViT→abnormal, CNN→normal after pretrained-init) is stable across seeds, not a seed-1 artifact — flag any seed where it flips
- [ ] (optional) paired significance test (DeLong on AUC, or bootstrap) between pretrained-init and baseline / OGM-GE
- [ ] Replace all seed=1 numbers in the draft abstract + CHANGELOG comparison table with mean ± std

**Step 2.5 — REBUILD CLEAN SPLIT (blocker for final numbers; found 2026-07-13)**

Data leakage confirmed: current `data/train.csv`+`data/val.csv` pool the PhysioNet
`training-a…f` folders PLUS the official `validation/` folder, which duplicates
recordings already in training. 85 recordings overlap train↔val (~12% of val),
identical `.npy` files; plus duplicate rows within train. Inflates all absolute
metrics (partly explains AUC 0.9668 vs paper 0.8608). Current 5-seed runs are being
finished ONLY to confirm the gradient-dominance hypothesis is seed-stable — those
numbers are PROVISIONAL and must not appear in the final paper.

Cleaning plan (do AFTER the current leaky-split 5-seed runs finish):
- [ ] Step A — investigate metadata: check whether any patient→recording mapping file exists in the data folders (subset `a` may have some); read-only, decides whether patient-disjoint is achievable
- [ ] Step B — source of truth: build the master list from `training-a…f` ONLY (one entry per physical recording, key = recording filename e.g. `a0001`); DROP all `validation/`-folder rows (they are the duplicate source)
- [ ] Step C — deduplicate: collapse to one row per unique recording ID (duplicates confirmed to point to identical `.npy`, so lossless). Removes the 85 train↔val overlaps + within-train dupes. Expect ~3240 unique recordings (not 3541)
- [ ] Step D — fresh split via `make_clean_split.py`: recording-disjoint (split the UNIQUE list), class-stratified (~77/23 preserved), ideally also stratified by subset a–f. Consider k-fold here for final rigor
- [ ] Step E — verify (critical): assert ZERO filename overlap AND zero `.npy` overlap between train/val; class ratios preserved both sides; print a verification report. Add a `test_clean_split.py`
- [ ] Step F — regenerate CSVs in the SAME column format (`filename,label,filepath,npy_filepath`) as `data/train_clean.csv` / `data/val_clean.csv` (or k-fold files) so all training scripts run unchanged
- [ ] Step G — re-run the full multi-seed pipeline (Step 2) on the clean split → these become the REAL publishable numbers
- [ ] Step H — compare clean-split baseline AUC vs the leaky 0.9668 to quantify how much was leakage; expect a modest drop (~0.90–0.93 range)
- [ ] Honesty note for the paper: state "recording-disjoint, class/subset-stratified; patient-level disjointness not guaranteed for all subsets due to missing PhysioNet metadata" — the accepted standard for this dataset

**Step 3 — explainability / the "why" (the closing section; rests on Step 2 being stable)**

Minimal 2-figure plan (supporting evidence, not core contribution). Build one at a
time, judge whether each clearly tells the story, drop anything confusing. Both run
on existing checkpoints — no retraining.

- [ ] **Figure 1 (do first — highest impact, easiest to read): t-SNE of ViT features, baseline vs pretrained-init.** Extract ViT branch features (192-dim, pre-head) on the val set for both models; project to 2D with t-SNE, color by class. Expected story: baseline = one mixed/overlapping blob (ViT can't separate classes); pretrained-init = two clean clusters (ViT now discriminates). Visually proves the central claim with zero domain expertise required.
- [ ] **Figure 2: spectrogram attention heatmap on the ViT (the clinical "where").** Attention rollout (or GradCAM) over spectrogram patches for representative normal vs abnormal samples; check whether abnormal attention concentrates on murmur bands (~100–500 Hz) / irregular S1–S2 timing. The clinical hook.
- [ ] Assemble the 2-figure panel + write the "what each branch does and why" section, linking branch specialization to cardiac-acoustic domain knowledge.

Optional add-ons (only if a reviewer asks or time permits — do NOT front-load):
- [ ] Waveform input-gradient saliency / SmoothGrad on the CNN branch (does it lock onto regular S1–S2 rhythm/envelope in normal cases?)
- [ ] SHAP on the fusion head treating [f_spec; f_wave] as features → game-theoretic per-branch importance
- [ ] CKA between f_spec and f_wave → single-number complementarity measure (links to §3.1.5)

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

### 2.-1 Gradient-Dominance Generalization to a 2nd Dataset (future paper idea — 2026-07-14)

Idea: strengthen the gradient-dominance findings (Direction 3.1) by replicating them
on a SECOND dataset with the SAME dual-branch ViT+CNN architecture. Goal: show the
phenomenon is architectural (about optimization dynamics), not a PhysioNet/cardiac
quirk. If the ViT again dies in joint training and pretrained-init again revives it,
the finding generalizes → lifts the contribution. Likely a separate follow-up paper.

- [ ] Pick 2nd dataset — recommended **ICBHI 2017 lung sounds** (different pathology, still biomedical audio, keeps the clinical narrative; already needed for 2.1 transfer work). Alternatives: PASCAL heart sounds (weaker — same domain), ESC-50/UrbanSound8K (strongest generality but leaves biomedical framing).
- [ ] Preprocess into the same dual input (log-Mel spectrogram + raw waveform); reuse the ICBHI pipeline built for §2.1
- [ ] Run the full remedy pipeline on it: baseline + branch ablation (is ViT dead?), pretrained-init (does it revive?), gradient-flow trace
- [ ] Report whether gradient dominance + the pretrained-init fix replicate → cross-dataset generalization claim
- [ ] NOTE: do this only AFTER the PhysioNet clean-split rerun (Step 2.5); don't build on the leaky methodology. Bonus: the ICBHI preprocessing doubles as the input for the §2.1 transfer experiments.

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
- [ ] Gradient flow visualization on random-init joint training: log per-branch gradient norms (‖∇L/∇θ_spec‖ and ‖∇L/∇θ_wave‖) at every epoch; plot gradient norm curves over training to visually demonstrate CNN dominates early and ViT receives vanishing gradients — direct empirical proof of gradient dominance as the root cause
- [ ] Explainability — why does ViT specialize in abnormal and CNN in normal after pretrained init?
    - GradCAM on ViT branch: visualize attended spectrogram patches for normal vs abnormal samples — expect ViT to focus on murmur frequency bands (100–500 Hz) and irregular S1/S2 timing in abnormal cases
    - Waveform saliency on CNN branch: compute input gradients w.r.t. waveform for normal vs abnormal — expect CNN to focus on regular heartbeat envelope/rhythm in normal cases
    - Compare attention maps pre vs post pretrained-init to show the ViT learned meaningful frequency features only after proper training
    - This adds an explainability/interpretability dimension to the paper — answers the "why" behind branch specialization
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
