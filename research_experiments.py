"""Reproducible experiment runner for the research roadmap."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("[cmd]", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def run_compare(args):
    outdir = Path(args.outdir)
    controllers = args.controllers.split(",")
    for controller in controllers:
        run_dir = outdir / controller
        run_command([
            sys.executable,
            "train.py",
            "--phase",
            "imitation",
            "--controller",
            controller,
            "--num-traj",
            str(args.num_traj),
            "--seq-steps",
            str(args.seq_steps),
            "--epochs",
            str(args.epochs),
            "--outdir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ])
        run_command([
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            str(run_dir / "best_imitation.pt"),
            "--missions",
            str(args.missions),
            "--seq-steps",
            str(args.seq_steps),
            "--seed",
            str(args.seed),
        ])


def run_sample_efficiency(args):
    outdir = Path(args.outdir)
    for num_traj in args.sample_counts:
        for controller in args.controllers.split(","):
            run_dir = outdir / f"{controller}_n{num_traj}"
            run_command([
                sys.executable,
                "train.py",
                "--phase",
                "imitation",
                "--controller",
                controller,
                "--num-traj",
                str(num_traj),
                "--seq-steps",
                str(args.seq_steps),
                "--epochs",
                str(args.epochs),
                "--outdir",
                str(run_dir),
                "--seed",
                str(args.seed),
            ])
            run_command([
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                str(run_dir / "best_imitation.pt"),
                "--missions",
                str(args.missions),
                "--seq-steps",
                str(args.seq_steps),
                "--seed",
                str(args.seed),
            ])


def run_robustness(args):
    checkpoint = Path(args.checkpoint)
    for profile in ["nominal", "wide-init", "constraint-tight"]:
        run_command([
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--missions",
            str(args.missions),
            "--seq-steps",
            str(args.seq_steps),
            "--seed",
            str(args.seed),
            "--robustness-profile",
            profile,
        ])


def run_validate(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_command([
        sys.executable,
        "evaluate.py",
        "--expert-only",
        "--missions",
        str(args.missions),
        "--seq-steps",
        str(args.seq_steps),
        "--seed",
        str(args.seed),
    ])
    with open(outdir / "validation_manifest.json", "w") as f:
        json.dump({"validated_with": "evaluate.py --expert-only"}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run NNCS-Mamba research experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", default="runs/research")
    common.add_argument("--controllers", default="mlp,gru,mamba")
    common.add_argument("--num-traj", type=int, default=1024)
    common.add_argument("--seq-steps", type=int, default=300)
    common.add_argument("--epochs", type=int, default=8)
    common.add_argument("--missions", type=int, default=20)
    common.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("compare", parents=[common])

    sample = subparsers.add_parser("sample-efficiency", parents=[common])
    sample.add_argument("--sample-counts", type=int, nargs="+", default=[128, 512, 2048])

    robust = subparsers.add_parser("robustness")
    robust.add_argument("--checkpoint", required=True)
    robust.add_argument("--seq-steps", type=int, default=300)
    robust.add_argument("--missions", type=int, default=20)
    robust.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--outdir", default="runs/validation")
    validate.add_argument("--seq-steps", type=int, default=300)
    validate.add_argument("--missions", type=int, default=20)
    validate.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "compare":
        run_compare(args)
    elif args.command == "sample-efficiency":
        run_sample_efficiency(args)
    elif args.command == "robustness":
        run_robustness(args)
    elif args.command == "validate":
        run_validate(args)


if __name__ == "__main__":
    main()
