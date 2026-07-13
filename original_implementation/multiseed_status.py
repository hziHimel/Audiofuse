"""
Multi-seed pipeline status/planner (Direction 3.1, Step 2).

Read-only: scans the filesystem for completed artifacts of the 5-seed
comparative study and reports a done/pending grid plus the exact command for
the next pending step. Runs NOTHING — safe to call anytime to resume after a
break. Steps are atomic (one training or one ablation each) and detected by the
persistent seed-suffixed checkpoint / ablation CSV they produce.

Usage:
    python multiseed_status.py            # LEAKY split (data/train.csv), outputs/...
    python multiseed_status.py --clean    # CLEAN split (data/train_clean.csv), outputs/clean/...
    python multiseed_status.py --seed 2   # just seed 2's next command
"""

import os
import argparse

SEEDS = [1, 2, 3, 4, 5]


def build_stages(clean: bool):
    """Build the 10-stage pipeline for either the leaky or clean split.

    clean=False → original split (data/train.csv, outputs/…) — our seed 1–3 runs.
    clean=True  → deduplicated split (data/train_clean.csv, outputs/clean/…) — the
                  real publishable 5-seed runs and the teammate reproduction.
    """
    train = "data/train_clean.csv" if clean else "data/train.csv"
    val   = "data/val_clean.csv"   if clean else "data/val.csv"
    o     = "outputs/clean/" if clean else "outputs/"          # output prefix
    data  = f"--train_csv {train} --val_csv {val}"

    return [
        ("baseline",
         f"{o}pytorch/best_seed{{n}}.pt",
         f"python train_pytorch.py {data} --seeds {{n}} --output_dir {o}pytorch/"),
        ("waveonly",
         f"{o}pytorch_waveonly/best_seed{{n}}.pt",
         f"python train_pytorch_waveonly.py {data} --seeds {{n}} --output_dir {o}pytorch_waveonly/"),
        ("speconly",
         f"{o}pytorch_speconly/best_seed{{n}}.pt",
         f"python train_pytorch_speconly.py {data} --seeds {{n}} --output_dir {o}pytorch_speconly/"),
        ("pretrained_init",
         f"{o}pytorch_pretrained_init/best_seed{{n}}.pt",
         f"python train_pytorch_pretrained_init.py {data} "
         f"--spec_ckpt {o}pytorch_speconly/best_seed{{n}}.pt "
         f"--wave_ckpt {o}pytorch_waveonly/best_seed{{n}}.pt "
         f"--seeds {{n}} --output_dir {o}pytorch_pretrained_init/"),
        ("ogm",
         f"{o}pytorch_ogm/best_seed{{n}}.pt",
         f"python train_pytorch_ogm.py {data} --alpha 0.5 --seed {{n}} --output_dir {o}pytorch_ogm/"),
        ("moddrop",
         f"{o}pytorch_moddrop/best_seed{{n}}.pt",
         f"python train_pytorch_moddrop.py {data} --p 0.5 --seed {{n}} --output_dir {o}pytorch_moddrop/"),
        ("ablate_baseline",
         f"{o}ablation_ms/baseline_seed{{n}}/ablation_results.csv",
         f"python branch_ablation.py --val_csv {val} "
         f"--checkpoint {o}pytorch/best_seed{{n}}.pt "
         f"--output_dir {o}ablation_ms/baseline_seed{{n}}/"),
        ("ablate_pretrained",
         f"{o}ablation_ms/pretrained_seed{{n}}/ablation_results.csv",
         f"python branch_ablation.py --val_csv {val} "
         f"--checkpoint {o}pytorch_pretrained_init/best_seed{{n}}.pt "
         f"--output_dir {o}ablation_ms/pretrained_seed{{n}}/"),
        ("ablate_ogm",
         f"{o}ablation_ms/ogm_seed{{n}}/ablation_results.csv",
         f"python branch_ablation.py --val_csv {val} "
         f"--checkpoint {o}pytorch_ogm/best_seed{{n}}.pt "
         f"--output_dir {o}ablation_ms/ogm_seed{{n}}/"),
        ("ablate_moddrop",
         f"{o}ablation_ms/moddrop_seed{{n}}/ablation_results.csv",
         f"python branch_ablation.py --val_csv {val} "
         f"--checkpoint {o}pytorch_moddrop/best_seed{{n}}.pt "
         f"--output_dir {o}ablation_ms/moddrop_seed{{n}}/"),
    ]


def is_done(stage_artifact: str, n: int) -> bool:
    return os.path.exists(stage_artifact.format(n=n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="only report this seed")
    parser.add_argument("--clean", action="store_true",
                        help="use the clean deduplicated split (data/*_clean.csv, outputs/clean/)")
    args = parser.parse_args()

    STAGES = build_stages(args.clean)
    print(f"[{'CLEAN split' if args.clean else 'LEAKY split'}]")

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
