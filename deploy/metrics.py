"""The metric framework of Section V, as executable formulas.

Every function maps measured inputs + the vendor constants (deploy/constants.py)
to a reported quantity, tagged with its manuscript equation number.  Pure Python /
numpy so it runs on host or target, and so the results tables can be regenerated
deterministically from raw measurements.
"""

from __future__ import annotations

from . import constants as C


# ---- Timing --------------------------------------------------------------
def fmax_hz(target_period_ns: float, wns_ns: float) -> float:
    """f_max = 1 / (T - WNS)   (Eq. (9)). WNS positive when timing is met."""
    return 1e9 / (target_period_ns - wns_ns)


# ---- Area ----------------------------------------------------------------
def utilization(used: dict) -> dict:
    """U_r = used_r / avail_r * 100%   (Eq. (10))."""
    out = {}
    for r, tot in C.RESOURCE_TOTALS.items():
        if r in used and used[r] is not None:
            out[r] = {"used": used[r], "avail": tot, "pct": 100.0 * used[r] / tot}
    return out


# ---- Throughput and roofline ---------------------------------------------
def peak_tops() -> float:
    """P_peak (Eq. (3))."""
    return C.PEAK_TOPS


def effective_tops(workload_ops: float, l_dpu_s: float) -> float:
    """P_eff = W / L_DPU   (Eq. (12)), in TOPS."""
    return workload_ops / l_dpu_s / 1e12


def efficiency(p_eff_tops: float) -> float:
    """eta = P_eff / P_peak   (Eq. (13))."""
    return p_eff_tops / C.PEAK_TOPS


def arithmetic_intensity(workload_ops: float, traffic_bytes: float) -> float:
    """I = W / Q   (Eq. (14)), ops/byte."""
    return workload_ops / traffic_bytes


def roofline_tops(intensity_op_byte: float) -> float:
    """P_roof = min(P_peak, beta_DDR * I)   (Eq. (15)), in TOPS."""
    mem_bound_tops = C.DDR_BW_GBS * 1e9 * intensity_op_byte / 1e12
    return min(C.PEAK_TOPS, mem_bound_tops)


def ridge_point() -> float:
    """I* = P_peak / beta_DDR (op/byte) -- roofline ridge (Fig. 6)."""
    return (C.PEAK_TOPS * 1e12) / (C.DDR_BW_GBS * 1e9)


# ---- Latency and pipeline throughput -------------------------------------
def e2e_latency_s(l_pre, l_dpu, l_post, l_ps_gs=0.0) -> float:
    """L_e2e = L_pre + L_DPU + L_post (+ L_PS_GS for R3)   (Eq. (16))."""
    return l_pre + l_dpu + l_post + l_ps_gs


def fps_single(l_e2e_s: float) -> float:
    """FPS_single = 1 / L_e2e   (Eq. (17))."""
    return 1.0 / l_e2e_s


def fps_pipe(stage_latencies_s) -> float:
    """FPS_pipe ~= 1 / max_s L_s   (Eq. (17)), steady-state pipeline."""
    return 1.0 / max(stage_latencies_s)


# ---- Power and energy -----------------------------------------------------
def avg_power_w(samples) -> float:
    """P_avg = (1/N) sum_n sum_i V_i[n] I_i[n]   (Eq. (18)).

    ``samples`` is an iterable of per-timestep total watts (already summed over
    rails), or of dicts {rail: (V, I)}.
    """
    tot, n = 0.0, 0
    for s in samples:
        if isinstance(s, dict):
            tot += sum(v * i for (v, i) in s.values())
        else:
            tot += float(s)
        n += 1
    return tot / max(n, 1)


def energy_per_inference_mj(p_load_w: float, fps_pipe_val: float) -> float:
    """E = P_load / FPS_pipe   (Eq. (19)), in millijoules per frame."""
    return 1000.0 * p_load_w / fps_pipe_val


def fps_per_watt(fps_pipe_val: float, p_load_w: float) -> float:
    """FPS/W   (Eq. (20))."""
    return fps_pipe_val / p_load_w


def gops_per_watt(p_eff_tops: float, p_load_w: float) -> float:
    """GOPS/W   (Eq. (20))."""
    return p_eff_tops * 1e3 / p_load_w


# ---- Accuracy deltas ------------------------------------------------------
def delta_q(acc_fp32: float, acc_int8: float) -> float:
    """Delta_q = Acc_FP32 - Acc_INT8   (Eq. (8))."""
    return acc_fp32 - acc_int8


def delta_fold(acc_full: float, acc_folded: float) -> float:
    """Delta_fold = Acc(full) - Acc(folded)   (Eq. (6))."""
    return acc_full - acc_folded


def model_workload_ops(per_layer):
    """W = sum_l 2 Kh Kw Cin Cout Hout Wout   (Eq. (11)).

    ``per_layer`` is an iterable of dicts with keys kh, kw, cin, cout, hout, wout
    (from the compiler's per-layer report).
    """
    w = 0
    for L in per_layer:
        w += 2 * L["kh"] * L["kw"] * L["cin"] * L["cout"] * L["hout"] * L["wout"]
    return w


if __name__ == "__main__":
    # Self-check against the manuscript's stated constants.
    print("P_peak (TOPS):", round(peak_tops(), 3), "(expect 1.229)")
    print("beta_DDR (GB/s):", C.DDR_BW_GBS, "(expect 19.2)")
    print("ridge I* (op/byte):", round(ridge_point(), 2))
    print("f_max @ WNS=0.3ns, T=3.333ns:", round(fmax_hz(3.333, 0.3) / 1e6, 1), "MHz")
