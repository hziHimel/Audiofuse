# AudioFuse Extension — Development Guide

## Overview

This project extends **AudioFuse**, a dual-branch deep learning architecture (ViT + 1D-CNN) for
heart sound classification on the PhysioNet 2016 dataset. The base reproduction is in
`original_implementation/`. All extension experiments live alongside or within that directory.

**Research goals:**
1. **Direction 1 — Improve heart sound detection**: better training strategies (threshold optimization,
   focal loss, k-fold CV), preprocessing (MFCC, gammatone, segmentation), and architecture changes
   (attention pooling, gated fusion, residual CNN).
2. **Direction 2 — Cross-modal transfer learning**: freeze the PhysioNet-pretrained AudioFuse encoder
   and evaluate on ECG (MIT-BIH) and lung sounds (ICBHI 2017). Test the hypothesis that cardiac
   acoustic pretraining captures transferable temporal-periodicity representations.
3. **Direction 3 — Fusion analysis**: understand how branches contribute (ablation, GradCAM, SHAP),
   then improve the fusion mechanism (gated, cross-attention, bilinear, contrastive loss).

See `research_directions.md` for full detail on all three directions.

## Quick Reference

- **Base implementation**: `original_implementation/`
- **Training (PyTorch)**: `original_implementation/train_pytorch.py`
- **Training (Keras)**: `original_implementation/train_keras.py`
- **Experiment tasks**: `TODO.md`
- **Progress log**: `CHANGELOG.md`
- **Research directions**: `research_directions.md`

## Orientation (start here each session)

1. Read `CHANGELOG.md` — see what's done and what the current open task is.
2. Read `TODO.md` — pick the next unchecked item that's unblocked.
3. Run existing tests before changing anything: `python -m pytest test_*.py -v`
4. Implement → test → update `CHANGELOG.md` → mark item done in `TODO.md`.

## Environment

```bash
# Activate the project venv (Python 3.11, has torch, librosa, sklearn, etc.)
source original_implementation/venv311/bin/activate

# Run tests
python -m pytest original_implementation/test_*.py -v

# Train (requires GPU + preprocessed data in original_implementation/data/)
python original_implementation/train_pytorch.py \
    --train_csv original_implementation/data/train.csv \
    --val_csv   original_implementation/data/val.csv \
    --seeds 1
```

Device priority: MPS (Apple Silicon) → CUDA → CPU. The model runs on CPU for unit tests.

## Filesystem rules

- **Project root for searches**: `Experiments/` and its subdirectories only.
- Never scan outside the project directory.
- Use `grep` or `Grep` for content search, not `find` with broad paths.

## Principles for autonomous development

### 1. Verify before and after

The scikit-learn metrics (AUC, F1, MCC) on the val set are our oracle — like CLASS in the
CLAX project. Every change must show a number that either improves or matches the baseline.

- Never commit code that breaks an existing passing test.
- Write a test reproducing a bug before fixing it.
- Every new utility (loss, augmentation, metric) must have a `test_*.py` before use in training.

### 2. Concise test output

- Tests print at most 5–10 lines on success, ~20 on failure.
- Never dump full arrays; print max error, index, and pass/fail summary.
- Good: `FAILED test_threshold_sweep::test_optimal - opt F1=0.712 < baseline 0.846`

### 3. Keep CHANGELOG.md current

CHANGELOG.md is the shared memory across sessions. Update it after every meaningful unit of work:
- Check off completed items with dates.
- Note what worked and what didn't.
- Record failed approaches so they aren't re-attempted.

### 4. Small, testable changes

- One change per commit (one loss function, one augmentation, one architecture variant).
- Each change: implement → unit test → run training (or verify logic without full run) → log result.
- If a full GPU training run is needed, note it as "pending GPU run" in CHANGELOG.md.

### 5. Parallel agents (use liberally)

Agent teams are enabled. Use them to parallelize independent experiments:

- **Training runs on different seeds** → one agent per seed
- **Preprocessing variants** → one agent generates MFCC, another gammatone simultaneously
- **Competing hypotheses** → one agent investigates focal loss, another investigates threshold sweep

When working in parallel, note your task in CHANGELOG.md as `IN PROGRESS: <task> (@agent-N)`.

### 6. Structure work for parallelism

Easy to parallelize (independent):
- Loss function variants (focal, label smoothing) — different files, no shared state
- Preprocessing alternatives (MFCC, gammatone) — operate on independent output dirs
- Branch ablations (spec-only, wave-only) — already separate training scripts

Hard to parallelize (shared state):
- Cross-validation wrappers that write to the same results.csv
- Any change to `train_pytorch.py` itself (coordinate via separate feature branches)

### 7. Document for the next session

Every new function/class needs a one-line docstring explaining:
- What it does and why (not just what — the name already says what).
- Any non-obvious numerical choices (e.g., why α=0.25 for focal loss).

---

## Coding conventions

- **Python 3.11**, PyTorch 2.x. Match the venv in `original_implementation/venv311/`.
- **Type hints** on all new public functions.
- No dead code, no commented-out experiments left in source files — use CHANGELOG.md instead.
- Match existing style in `train_pytorch.py` (no docstrings on private helpers, snake_case).
- New experiments go in new files (e.g., `train_pytorch_focal.py`, `train_pytorch_kfold.py`),
  not inline modifications to the base training script — except for utilities like `sweep_threshold`
  that are added to `train_pytorch.py` and used by all runs.

## Accuracy baselines (seed=1, fixed threshold=0.50)

| Framework | Accuracy | F1     | ROC-AUC | MCC    |
|-----------|----------|--------|---------|--------|
| PyTorch   | 0.9267   | 0.8462 | 0.9668  | 0.7990 |
| Keras     | 0.8858   | 0.7291 | 0.8983  | 0.6618 |

Any experiment result should be compared against the PyTorch seed=1 baseline as the reference.

## General Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Test suite

```bash
# Run all tests
python -m pytest original_implementation/test_*.py -v

# Run fast (smoke test, deterministic subsample where applicable)
python -m pytest original_implementation/test_*.py -v -k "not slow"
```

Current tests:
- `original_implementation/test_threshold_sweep.py` — 4 tests for sweep_threshold()
