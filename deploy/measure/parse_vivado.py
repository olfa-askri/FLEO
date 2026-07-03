"""Extract Table IV (utilization) and Table V (timing) inputs from Vivado reports.

Best-effort text parsers for:
  * ``report_utilization`` output  -> used LUT/FF/BRAM36/URAM/DSP  (Eq. (10))
  * ``report_timing_summary`` output -> worst negative slack (WNS)  (Eq. (9))

Vivado report formatting varies across versions; the parsers key on stable row
labels and fall back gracefully.  Verify the extracted numbers against the report
header before publishing.

Usage:
    python -m deploy.measure.parse_vivado --util post_impl_util.rpt \
        --timing timing_summary.rpt --out results/impl_r1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Row label -> canonical resource key. First integer in the row is "Used".
UTIL_ROWS = [
    (re.compile(r"\bCLB LUTs\b|\bSlice LUTs\b"), "LUT"),
    (re.compile(r"\bCLB Registers\b|\bSlice Registers\b|Register as Flip Flop"), "FF"),
    (re.compile(r"\bBlock RAM Tile\b"), "BRAM36"),
    (re.compile(r"\bURAM\b"), "URAM"),
    (re.compile(r"\bDSPs\b|\bDSP48"), "DSP"),
]


def parse_utilization(text: str) -> dict:
    used = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        for pat, key in UTIL_ROWS:
            if key in used:
                continue
            if pat.search(line):
                cells = [c.strip() for c in line.split("|")]
                for c in cells[2:]:  # skip the label cell(s)
                    m = re.fullmatch(r"\d+", c.replace(",", ""))
                    if m:
                        used[key] = int(c.replace(",", ""))
                        break
    return used


_WNS_RE = re.compile(r"WNS\s*\(ns\)?\s*[:|]?\s*(-?\d+\.\d+)", re.IGNORECASE)


def parse_timing(text: str) -> dict:
    """Return {'design_wns_ns': float, 'per_clock': {clk: wns}} best-effort."""
    out = {"design_wns_ns": None, "per_clock": {}}
    # Design Timing Summary: the first WNS number after the header.
    for m in _WNS_RE.finditer(text):
        out["design_wns_ns"] = float(m.group(1))
        break
    # Per-clock (Intra Clock Table rows: "clk_name  ... WNS ...") -- heuristic.
    for line in text.splitlines():
        m = re.search(r"(\w+clk\w*|clk_\w+|dpu\w*)\s+.*?(-?\d+\.\d+)", line)
        if m and "WNS" not in line:
            name, val = m.group(1), float(m.group(2))
            out["per_clock"].setdefault(name, val)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--util", help="report_utilization text file")
    ap.add_argument("--timing", help="report_timing_summary text file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ..metrics import utilization, fmax_hz
    from .. import constants as C

    result = {}
    if args.util:
        used = parse_utilization(Path(args.util).read_text(errors="ignore"))
        result["utilization"] = utilization(used)
    if args.timing:
        timing = parse_timing(Path(args.timing).read_text(errors="ignore"))
        result["timing"] = timing
        if timing["design_wns_ns"] is not None:
            result["fmax"] = {}
            for dom, spec in C.CLOCK_DOMAINS.items():
                result["fmax"][dom] = {
                    "target_MHz": spec["target_hz"] / 1e6,
                    "target_ns": round(spec["target_ns"], 3),
                    "wns_ns": timing["design_wns_ns"],
                    "fmax_MHz": round(fmax_hz(spec["target_ns"], timing["design_wns_ns"]) / 1e6, 1),
                }

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
