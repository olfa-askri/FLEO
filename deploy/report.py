"""Assemble the results tables (Section VII) from collected measurement JSON.

Consumes the per-route artifacts produced by the deployment kit:
  * impl_<route>.json    (parse_vivado)   -> Tables IV, V
  * latency_<route>.json (vart_pipeline)  -> Table VII
  * power_idle.json / power_<route>.json (power_ina226) -> Table VI
  * workload_<route>.json (compiler per-layer op count)  -> Eqs. (11)-(15)
  * accuracy JSON (scripts.evaluate / scripts.deltas)    -> Table VIII

Every derived quantity is computed with deploy/metrics.py so the printed tables
are reproducible from the raw inputs.  Missing inputs are rendered as the paper's
pending dashes, so the report is honest about what has and has not been measured.

Usage:
    python -m deploy.report --results-dir results --routes r1 r2 r3 \
        --out results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import constants as C
from . import metrics as M

DASH = "--"


def load(path: Path):
    if path and path.exists():
        return json.loads(path.read_text())
    return None


def fmt(x, nd=2):
    return DASH if x is None else f"{x:.{nd}f}"


def table_resources(impls):
    rows = ["### Table IV -- Post-implementation resource utilization (Eq. (10))",
            "",
            "| Resource | Avail (XCZU7EV) | " + " | ".join(f"{r} used / %" for r in impls) + " |",
            "|---|---|" + "|".join(["---"] * len(impls)) + "|"]
    for res, tot in C.RESOURCE_TOTALS.items():
        cells = []
        for r, data in impls.items():
            u = (data or {}).get("utilization", {}).get(res)
            cells.append(f"{u['used']} / {u['pct']:.1f}%" if u else DASH)
        rows.append(f"| {res} | {tot:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_timing(impls):
    rows = ["### Table V -- Timing closure per PL clock domain (Eq. (9))",
            "",
            "| Domain | Target f | Target T (ns) | " + " | ".join(f"{r}: WNS / fmax" for r in impls) + " |",
            "|---|---|---|" + "|".join(["---"] * len(impls)) + "|"]
    for dom, spec in C.CLOCK_DOMAINS.items():
        cells = []
        for r, data in impls.items():
            fm = (data or {}).get("fmax", {}).get(dom)
            cells.append(f"{fm['wns_ns']} / {fm['fmax_MHz']} MHz" if fm else DASH)
        rows.append(f"| {dom} | {spec['target_hz']/1e6:.0f} MHz | {spec['target_ns']:.3f} | "
                    + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_power_energy(powers, latencies, idle):
    rows = ["### Table VI -- Power and energy (Eqs. (18)-(20))",
            "",
            "| Config | P_idle (W) | P_load (W) | E (mJ/frame) | FPS/W |",
            "|---|---|---|---|---|"]
    p_idle = (idle or {}).get("P_avg_W")
    for r in powers:
        pw = powers[r]
        lat = latencies.get(r)
        p_load = (pw or {}).get("P_avg_W")
        fps = (lat or {}).get("fps_pipe_measured") or (lat or {}).get("fps_pipe_model")
        E = M.energy_per_inference_mj(p_load, fps) if (p_load and fps) else None
        fpsw = M.fps_per_watt(fps, p_load) if (p_load and fps) else None
        rows.append(f"| {r} | {fmt(p_idle)} | {fmt(p_load)} | {fmt(E)} | {fmt(fpsw)} |")
    return "\n".join(rows)


def table_latency(latencies, workloads):
    rows = ["### Table VII -- Latency decomposition and throughput (Eqs. (16)-(17),(12)-(13))",
            "",
            "| Config | L_pre | L_dpu | L_post | L_e2e (ms) | FPS_single | FPS_pipe | eta |",
            "|---|---|---|---|---|---|---|---|"]
    for r, lat in latencies.items():
        if not lat:
            rows.append(f"| {r} | " + " | ".join([DASH] * 7) + " |")
            continue
        eta = None
        w = workloads.get(r)
        if w and lat.get("L_dpu_ms"):
            p_eff = M.effective_tops(w["W_ops"], lat["L_dpu_ms"] / 1000.0)
            eta = M.efficiency(p_eff)
        rows.append(
            f"| {r} | {fmt(lat.get('L_pre_ms'))} | {fmt(lat.get('L_dpu_ms'))} | "
            f"{fmt(lat.get('L_post_ms'))} | {fmt(lat.get('L_e2e_ms'))} | "
            f"{fmt(lat.get('fps_single'))} | {fmt(lat.get('fps_pipe_measured') or lat.get('fps_pipe_model'))} | "
            f"{fmt(eta, 3)} |"
        )
    return "\n".join(rows)


def table_accuracy(acc):
    rows = ["### Table VIII -- Accuracy / macro-F1 by variant and precision (Eqs. (6),(8))",
            "",
            "| Variant | FER2013 FP32 | FER2013 INT8 | RAF-DB FP32 | RAF-DB INT8 |",
            "|---|---|---|---|---|"]
    if not acc:
        for v in ("Baseline", "Full (R3)", "Folded (R1)", "Householder (R2)", "Delta_fold"):
            rows.append(f"| {v} | {DASH} | {DASH} | {DASH} | {DASH} |")
    else:
        for v, cells in acc.items():
            rows.append(f"| {v} | " + " | ".join(fmt(c, 3) if isinstance(c, (int, float)) else DASH
                                                 for c in cells) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--routes", nargs="+", default=["r1", "r2", "r3"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    impls = {r: load(rd / f"impl_{r}.json") for r in args.routes}
    lats = {r: load(rd / f"latency_{r}.json") for r in args.routes}
    powers = {r: load(rd / f"power_{r}.json") for r in args.routes}
    workloads = {r: load(rd / f"workload_{r}.json") for r in args.routes}
    idle = load(rd / "power_idle.json")
    acc = load(rd / "accuracy_table.json")

    parts = [
        "# FLEO deployment results (ZCU104 / B4096)",
        "",
        f"Platform constants: P_peak = {C.PEAK_TOPS:.3f} TOPS, "
        f"beta_DDR = {C.DDR_BW_GBS} GB/s, ridge I* = {M.ridge_point():.1f} op/byte.",
        "",
        table_resources(impls), "",
        table_timing(impls), "",
        table_power_energy(powers, lats, idle), "",
        table_latency(lats, workloads), "",
        table_accuracy(acc), "",
        "_Dashes denote quantities pending real runs (Section VI)._",
    ]
    md = "\n".join(parts)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
