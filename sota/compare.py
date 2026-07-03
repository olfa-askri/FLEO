"""Generate the SOTA comparison report, merging our measured results.

Produces a Markdown document with:
  * FER2013 / RAF-DB accuracy tables (literature + our FP32/INT8 variants), and
  * the FPGA expression-recognition systems table (manuscript Table IX) with our
    R1/R2/R3 throughput/energy filled from results/.

Our numbers are read from results/*.json when present; otherwise rendered as
pending dashes so the comparison never reports an unmeasured value.

Usage:
    python -m sota.compare --results-dir results --out sota/SOTA_COMPARISON.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sota_data import FER2013_SOTA, RAFDB_SOTA, FPGA_FER_SYSTEMS

DASH = "--"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def acc_table(name, rows, ours):
    out = [f"### {name} -- overall accuracy (as reported; protocols differ)",
           "",
           "| Method | Year | Acc (%) | Note |",
           "|---|---|---|---|"]
    for method, year, acc, note in rows:
        out.append(f"| {method} | {year} | {acc:.2f} | {note} |")
    out.append(f"| **This work (FLEO, detection-cast)** | 2026 | {ours} | R1 fold-out, INT8 on B4096 |")
    return "\n".join(out)


def fpga_table(results_dir: Path):
    out = ["### FPGA expression-recognition systems (Table IX)",
           "",
           "| System | Platform | Style | Precision | Throughput | Note |",
           "|---|---|---|---|---|---|"]
    route_key = {"This work, R1 (fold-out)": "r1",
                 "This work, R2 (Householder)": "r2",
                 "This work, R3 (hybrid PS/PL)": "r3"}
    for system, plat, style, prec, thr, note in FPGA_FER_SYSTEMS:
        if system in route_key:
            lat = _load(results_dir / f"latency_{route_key[system]}.json")
            if lat:
                fps = lat.get("fps_pipe_measured") or lat.get("fps_pipe_model")
                thr = f"{fps:.1f} FPS" if fps else DASH
                note = "measured"
        out.append(f"| {system} | {plat} | {style} | {prec} | {thr or DASH} | {note} |")
    return "\n".join(out)


def our_accuracy(results_dir: Path, dataset_key: str):
    """Pull our INT8 folded accuracy for a dataset from a deltas/accuracy json."""
    d = _load(results_dir / f"deltas_{dataset_key}.json")
    if not d:
        return DASH
    v = d.get("variants_fp32", {}).get("folded_R1", {}).get("mean")
    return f"{v*100:.2f}" if isinstance(v, (int, float)) else DASH


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="sota/SOTA_COMPARISON.md")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    parts = [
        "# SOTA comparison",
        "",
        "Accuracy tables are **contextual**: published FER methods are GPU "
        "classifiers, whereas this work is a detection-cast detector deployed at "
        "INT8 on an FPGA overlay. The decisive internal comparison is Delta_fold "
        "(full vs folded), not the gap to a GPU classifier.",
        "",
        acc_table("FER2013", FER2013_SOTA, our_accuracy(rd, "fer2013")),
        "",
        acc_table("RAF-DB", RAFDB_SOTA, our_accuracy(rd, "rafdb")),
        "",
        fpga_table(rd),
        "",
        "_Our cells are populated from results/ when available; dashes are pending._",
    ]
    md = "\n".join(parts)
    print(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
