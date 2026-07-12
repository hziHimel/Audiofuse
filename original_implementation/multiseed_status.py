"""
Multi-seed pipeline status/planner (Direction 3.1, Step 2).

Read-only: scans the filesystem for completed artifacts of the 5-seed
comparative study and reports a done/pending grid plus the exact command for
the next pending step. Runs NOTHING — safe to call anytime to resume after a
break. Steps are atomic (one training or one ablation each) and detected by the
persistent seed-suffixed checkpoint / ablation CSV they produce.

Usage:
    python multiseed_status.py            # grid + next command
    python multiseed_status.py --seed 2   # just seed 2's next command
"""

import os
import argparse

SEEDS = [1, 2, 3, 4, 5]
DATA = "--train_csv data/train.csv --val_csv data/val.csv"

# Each stage: (name, artifact_path_template, command_template). Ordered by
# dependency: trainings before their ablations; branches before pretrained-init.
STAGES = [
    ("baseline",
     "outputs/pytorch/best_seed{n}.pt",
     f"python train_pytorch.py {DATA} --seeds {{n}} --output_dir outputs/pytorch/"),
    ("waveonly",
     "outputs/pytorch_waveonly/best_seed{n}.pt",
     f"python train_pytorch_waveonly.py {DATA} --seeds {{n}} --output_dir outputs/pytorch_waveonly/"),
    ("speconly",
     "outputs/pytorch_speconly/best_seed{n}.pt",
     f"python train_pytorch_speconly.py {DATA} --seeds {{n}} --output_dir outputs/pytorch_speconly/"),
    ("pretrained_init",
     "outputs/pytorch_pretrained_init/best_seed{n}.pt",
     f"python train_pytorch_pretrained_init.py {DATA} "
     "--spec_ckpt outputs/pytorch_speconly/best_seed{n}.pt "
     "--wave_ckpt outputs/pytorch_waveonly/best_seed{n}.pt "
     "--seeds {n} --output_dir outputs/pytorch_pretrained_init/"),
    ("ogm",
     "outputs/pytorch_ogm/best_seed{n}.pt",
     f"python train_pytorch_ogm.py {DATA} --alpha 0.5 --seed {{n}} --output_dir outputs/pytorch_ogm/"),
    ("moddrop",
     "outputs/pytorch_moddrop/best_seed{n}.pt",
     f"python train_pytorch_moddrop.py {DATA} --p 0.5 --seed {{n}} --output_dir outputs/pytorch_moddrop/"),
    ("ablate_baseline",
     "outputs/ablation_ms/baseline_seed{n}/ablation_results.csv",
     "python branch_ablation.py --val_csv data/val.csv "
     "--checkpoint outputs/pytorch/best_seed{n}.pt "
     "--output_dir outputs/ablation_ms/baseline_seed{n}/"),
    ("ablate_pretrained",
     "outputs/ablation_ms/pretrained_seed{n}/ablation_results.csv",
     "python branch_ablation.py --val_csv data/val.csv "
     "--checkpoint outputs/pytorch_pretrained_init/best_seed{n}.pt "
     "--output_dir outputs/ablation_ms/pretrained_seed{n}/"),
    ("ablate_ogm",
     "outputs/ablation_ms/ogm_seed{n}/ablation_results.csv",
     "python branch_ablation.py --val_csv data/val.csv "
     "--checkpoint outputs/pytorch_ogm/best_seed{n}.pt "
     "--output_dir outputs/ablation_ms/ogm_seed{n}/"),
    ("ablate_moddrop",
     "outputs/ablation_ms/moddrop_seed{n}/ablation_results.csv",
     "python branch_ablation.py --val_csv data/val.csv "
     "--checkpoint outputs/pytorch_moddrop/best_seed{n}.pt "
     "--output_dir outputs/ablation_ms/moddrop_seed{n}/"),
]


def is_done(stage_artifact: str, n: int) -> bool:
    return os.path.exists(stage_artifact.format(n=n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="only report this seed")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else SEEDS
    name_w = max(len(s[0]) for s in STAGES)

    print(f"\n{'stage':<{name_w}} | " + " ".join(f"s{n}" for n in seeds))
    print("-" * (name_w + 3 + 3 * len(seeds)))
    for name, artifact, _ in STAGES:
        marks = []
        for n in seeds:
            marks.append(" ✓" if is_done(artifact, n) else " ·")
        print(f"{name:<{name_w}} |" + "".join(marks))

    # find next pending step (seed-major: finish a seed before moving on)
    for n in seeds:
        for name, artifact, cmd in STAGES:
            if not is_done(artifact, n):
                print(f"\nNEXT: seed {n}, stage '{name}'")
                print(f"  {cmd.format(n=n)}")
                return
    print("\nAll steps complete for seeds:", seeds)


if __name__ == "__main__":
    main()
