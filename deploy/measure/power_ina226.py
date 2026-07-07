"""Sample ZCU104 board power from the INA226/PMBus rails (Eq. (18), Section V-E).

The ZCU104 exposes INA226 shunt monitors on its PMBus chain; on a PetaLinux image
they appear as hwmon devices (or via the sysfs iio/pmbus interface).  This sampler
reads the per-rail V*I products at a fixed rate and reports P_avg both at idle and
under load so the dynamic contribution is isolated.

Because the exact hwmon labels are board/image specific, pass --rails with the
sysfs power inputs you want summed, or use --autodiscover to sum every
``power*_input`` under /sys/class/hwmon.  Run one capture at idle and one under
sustained pipeline load, then feed both to deploy/report.py.

Usage (on target):
    # background sampler for the duration of a timed run
    python3 power_ina226.py --hz 10 --seconds 120 --tag load \
        --out /home/root/results/power_load.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path


def autodiscover_power_inputs():
    """Return sysfs 'power*_input' files (microwatts) across all hwmon devices."""
    return sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))


def discover_vi_rails():
    """Return (in_V_file, curr_A_file, label) triples for V*I computation."""
    rails = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = _read_str(Path(hw) / "name")
        for vin in sorted(glob.glob(f"{hw}/in*_input")):
            idx = Path(vin).name.replace("in", "").replace("_input", "")
            curr = Path(hw) / f"curr{idx}_input"
            if curr.exists():
                rails.append((vin, str(curr), f"{name}:rail{idx}"))
    return rails


def _read_str(p):
    try:
        return Path(p).read_text().strip()
    except Exception:
        return "?"


def _read_float(p):
    try:
        return float(Path(p).read_text().strip())
    except Exception:
        return 0.0


def sample_total_watts(power_inputs, vi_rails):
    """One instantaneous total-power sample in watts."""
    total = 0.0
    per_rail = {}
    for pf in power_inputs:
        w = _read_float(pf) / 1e6  # microwatt -> watt
        total += w
        per_rail[pf] = w
    for vin, curr, label in vi_rails:
        v = _read_float(vin) / 1e3   # millivolt -> volt
        a = _read_float(curr) / 1e3  # milliamp -> amp
        w = v * a
        total += w
        per_rail[label] = w
    return total, per_rail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hz", type=float, default=10.0, help="sampling rate (Section VI-B: 10 Hz)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--rails", nargs="*", default=None, help="explicit power*_input files")
    ap.add_argument("--autodiscover", action="store_true")
    ap.add_argument("--tag", default="load", choices=["idle", "load"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Rail selection: explicit --rails wins; otherwise autodiscover power*_input
    # files; if the board exposes no power*_input, fall back to summing V*I over
    # in*/curr* rail pairs.  (--autodiscover is the default and kept for clarity.)
    if args.rails:
        power_inputs = args.rails
    else:
        power_inputs = autodiscover_power_inputs()
    vi_rails = [] if power_inputs else discover_vi_rails()
    if not power_inputs and not vi_rails:
        raise SystemExit("No INA226 rails found under /sys/class/hwmon. Pass --rails explicitly.")

    period = 1.0 / args.hz
    n = int(args.seconds * args.hz)
    samples = []
    t0 = time.perf_counter()
    for _ in range(n):
        w, _pr = sample_total_watts(power_inputs, vi_rails)
        samples.append(w)
        dt = period - ((time.perf_counter() - t0) % period)
        time.sleep(max(0.0, dt))

    from ..metrics import avg_power_w

    p_avg = avg_power_w(samples)
    report = {
        "tag": args.tag,
        "hz": args.hz,
        "n": len(samples),
        "rails": power_inputs or [r[2] for r in vi_rails],
        "P_avg_W": p_avg,
        "P_min_W": min(samples),
        "P_max_W": max(samples),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
