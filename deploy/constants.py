"""Vendor-documented platform constants (Table I) and B4096 geometry (Fig. 2).

Every figure here is a datasheet/reference-design constant, not a measurement --
these are the numbers the metric framework (Section V) plugs measured quantities
into.  Kept in one place so the paper's tables and the on-target scripts agree.
"""

# ---- ZCU104 / XCZU7EV-2FFVC1156 (Table I) ---------------------------------
DEVICE = "XCZU7EV-2FFVC1156"
BOARD = "ZCU104"

RESOURCE_TOTALS = {          # denominators for utilization U_r (Eq. (10))
    "LUT": 230_400,
    "FF": 460_800,
    "BRAM36": 312,
    "URAM": 96,
    "DSP": 1_728,
}

# ---- B4096 DPUCZDX8G configuration (Fig. 2, Eqs. (2)-(3)) ------------------
PP = 8                        # pixel parallelism
ICP = 16                      # input-channel parallelism
OCP = 16                      # output-channel parallelism
PEAK_OPS_PER_CYCLE = 2 * PP * ICP * OCP          # = 4096  (Eq. (2))
F_CORE_HZ = 300e6             # DPU core clock (reference operating point)
F_DSP_HZ = 600e6             # DSP cascade domain (2 f_core, DDR option)
PEAK_TOPS = PEAK_OPS_PER_CYCLE * F_CORE_HZ / 1e12  # = 1.229 TOPS (Eq. (3))

# ---- DDR4 memory (Eq. (1)) ------------------------------------------------
DDR_MT_S = 2400               # DDR4-2400 mega-transfers/s
DDR_BUS_BYTES = 8             # 64-bit channel
DDR_BW_GBS = DDR_MT_S * DDR_BUS_BYTES / 1000.0     # = 19.2 GB/s (Eq. (1))

# ---- timing targets (Table V) ---------------------------------------------
CLOCK_DOMAINS = {
    "dpu_core":   {"target_hz": F_CORE_HZ, "target_ns": 1e9 / F_CORE_HZ},   # 3.333 ns
    "dsp_cascade": {"target_hz": F_DSP_HZ, "target_ns": 1e9 / F_DSP_HZ},    # 1.667 ns
}


def summary():
    return {
        "device": DEVICE,
        "board": BOARD,
        "peak_ops_per_cycle": PEAK_OPS_PER_CYCLE,
        "f_core_MHz": F_CORE_HZ / 1e6,
        "peak_TOPS": round(PEAK_TOPS, 3),
        "ddr_bw_GBps": DDR_BW_GBS,
        "resource_totals": RESOURCE_TOTALS,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=2))
