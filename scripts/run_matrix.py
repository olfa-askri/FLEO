"""Drive the full host-side experiment matrix (Section VI-C) end to end.

Crosses route x precision x threading is completed on the target; this script does
the host half: train (baseline + FLEO, N seeds), export the three routes, compute
Delta_fold / accuracy per dataset, and count workload for the roofline.  It shells
out to the other scripts so each step stays independently runnable.

Example:
    python -m scripts.run_matrix --data datasets/fer2013/data.yaml --dataset fer2013 \
        --seeds 0 1 2 --epochs 100 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = [sys.executable, "-m"]


def run(mod, *args):
    cmd = PY + [mod, *map(str, args)]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_ckpt(project: str, name: str) -> str:
    """Locate <name>/weights/best.pt, tolerating ultralytics' save-dir nesting."""
    direct = Path(project) / name / "weights" / "best.pt"
    if direct.exists():
        return str(direct)
    # Fall back to a search (handles runs/detect/... nesting on relative projects).
    roots = [Path(project), Path(project).parent, Path.cwd()]
    hits: list[Path] = []
    for root in roots:
        if root.exists():
            hits += list(root.rglob(f"{name}/weights/best.pt"))
    if not hits:
        raise SystemExit(
            f"Could not find trained checkpoint for '{name}'. Looked under "
            f"{[str(r) for r in roots]}. Did training finish?"
        )
    return str(max(hits, key=lambda p: p.stat().st_mtime))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/fleo")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    # Match train.py: absolute project so checkpoints land at <project>/<name>.
    args.project = str(Path(args.project).resolve())
    Path("results").mkdir(exist_ok=True)
    seeds = [str(s) for s in args.seeds]

    # 1) Train baseline + FLEO across seeds.
    if not args.skip_baseline:
        run("scripts.train", "--data", args.data, "--variant", "baseline",
            "--epochs", args.epochs, "--imgsz", args.imgsz, "--batch", args.batch,
            "--device", args.device, "--project", args.project, "--seeds", *seeds)
    run("scripts.train", "--data", args.data, "--variant", "fleo",
        "--epochs", args.epochs, "--imgsz", args.imgsz, "--batch", args.batch,
        "--device", args.device, "--project", args.project, "--seeds", *seeds)

    # 2) Export the three routes from the seed-0 checkpoint (representative).
    ckpt = find_ckpt(args.project, f"fleo_seed{args.seeds[0]}")
    print(f"[run_matrix] using checkpoint {ckpt}")
    for route in ("r1", "r2", "r3"):
        run("scripts.export", "--weights", ckpt, "--route", route,
            "--imgsz", args.imgsz, "--verify")
        run("deploy.measure.workload", "--weights", f"export/{route}.pt",
            "--route", route, "--imgsz", args.imgsz,
            "--out", f"results/workload_{route}.json")

    # 3) Delta_fold / accuracy across seeds.
    weights = [find_ckpt(args.project, f"fleo_seed{s}") for s in args.seeds]
    run("scripts.deltas", "--data", args.data, "--dataset", args.dataset,
        "--imgsz", args.imgsz, "--device", args.device,
        "--weights", *weights, "--out", f"results/deltas_{args.dataset}.json")

    print("\nHost matrix complete. Next: quantize + compile + run on the ZCU104 "
          "(see deploy/ and README).")


if __name__ == "__main__":
    main()
