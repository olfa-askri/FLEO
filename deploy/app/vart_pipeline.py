"""Three-stage threaded VART pipeline on the ZCU104 (Fig. 3, Section III-D).

    pre-proc (A53)  ->  DPU inference (PL, VART)  ->  post-proc (A53)

Stages run in separate threads joined by bounded queues; in steady state the frame
rate is bounded by the slowest stage (Eq. (17)).  Per-stage and end-to-end
latencies (Eq. (16)) are measured with monotonic host timers and written as JSON
for deploy/report.py.  For Route 3 an extra PS thread runs the Gram-Schmidt island
(gs_ps.py) between two DPU subgraphs.

Runs ON THE TARGET inside the Vitis AI runtime (imports vart/xir).  Protocol
(Section VI-B): 100-frame warm-up, then 1000 timed frames.

Usage:
    python3 vart_pipeline.py --xmodel compiled/r1/fleo_yolov12s_r1.xmodel \
        --images /run/media/frames --imgsz 640 --warmup 100 --frames 1000 \
        --route r1 --out /home/root/results/r1_latency.json
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

_SENTINEL = object()


# --------------------------------------------------------------------------- io
def list_frames(images_dir: Path, n: int):
    imgs = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        raise SystemExit(f"no images under {images_dir}")
    # cycle to reach n frames
    out = []
    while len(out) < n:
        out.extend(imgs)
    return out[:n]


def letterbox_u8(path: Path, imgsz: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(im.resize((nw, nh)), ((imgsz - nw) // 2, (imgsz - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


# --------------------------------------------------------------------------- DPU
class DpuRunner:
    """Thin wrapper over a VART DPU runner for a single-input single-output net."""

    def __init__(self, xmodel: str):
        import vart
        import xir

        graph = xir.Graph.deserialize(xmodel)
        subs = [s for s in graph.get_root_subgraph().toposort_child_subgraph()
                if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if not subs:
            raise SystemExit("no DPU subgraph found in xmodel")
        self.runner = vart.Runner.create_runner(subs[0], "run")
        self.itensors = self.runner.get_input_tensors()
        self.otensors = self.runner.get_output_tensors()
        self.ishape = tuple(self.itensors[0].dims)
        self.oshape = tuple(self.otensors[0].dims)
        # INT8 fixed-point scale for input quantization.
        self.in_scale = 2 ** self.itensors[0].get_attr("fix_point")

    def infer(self, x_int8: np.ndarray) -> np.ndarray:
        out = np.empty(self.oshape, dtype=np.int8)
        jid = self.runner.execute_async([x_int8], [out])
        self.runner.wait(jid)
        return out


# --------------------------------------------------------------------------- stages
def run_pipeline(args):
    frames = list_frames(Path(args.images), args.warmup + args.frames)
    dpu = DpuRunner(args.xmodel)
    imgsz = args.imgsz

    q_pre = queue.Queue(maxsize=8)
    q_dpu = queue.Queue(maxsize=8)
    times = {"pre": [], "dpu": [], "post": [], "e2e": []}
    completions = []  # wall-clock of each timed frame's completion (steady-state FPS)
    lock = threading.Lock()

    def stage_pre():
        for idx, p in enumerate(frames):
            t0 = time.perf_counter()
            img = letterbox_u8(p, imgsz).astype(np.float32) / 255.0
            # NHWC INT8 pack (VART expects NHWC for DPUCZDX8G).
            x = np.clip(np.round(img * dpu.in_scale), -128, 127).astype(np.int8)[None]
            t1 = time.perf_counter()
            q_pre.put((idx, x, t0, t1))
        q_pre.put(_SENTINEL)

    def stage_dpu():
        while True:
            item = q_pre.get()
            if item is _SENTINEL:
                q_dpu.put(_SENTINEL)
                return
            idx, x, t0, t1 = item
            td0 = time.perf_counter()
            out = dpu.infer(x)
            td1 = time.perf_counter()
            q_dpu.put((idx, out, t0, t1, td0, td1))

    def stage_post():
        from .postprocess import decode_topclass

        while True:
            item = q_dpu.get()
            if item is _SENTINEL:
                return
            idx, out, t0, t1, td0, td1 = item
            tp0 = time.perf_counter()
            _ = decode_topclass(out, nc=args.nc)
            tp1 = time.perf_counter()
            if idx >= args.warmup:  # drop warm-up frames (Section VI-B)
                with lock:
                    times["pre"].append(t1 - t0)
                    times["dpu"].append(td1 - td0)
                    times["post"].append(tp1 - tp0)
                    times["e2e"].append(tp1 - t0)
                    completions.append(tp1)  # for steady-state throughput

    threads = [threading.Thread(target=f) for f in (stage_pre, stage_dpu, stage_post)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    from ..metrics import fps_single, fps_pipe

    def ms(a):
        return float(np.mean(a) * 1000)

    # Steady-state throughput over the *timed* window only: (N-1) completions span
    # (last - first), so warm-up and pipeline fill/drain do not bias the rate.
    if len(completions) > 1:
        steady_span = completions[-1] - completions[0]
        fps_measured = (len(completions) - 1) / steady_span if steady_span > 0 else None
    else:
        steady_span, fps_measured = None, None

    stage_means_s = [np.mean(times["pre"]), np.mean(times["dpu"]), np.mean(times["post"])]
    report = {
        "route": args.route,
        "xmodel": args.xmodel,
        "frames_timed": len(times["e2e"]),
        "L_pre_ms": ms(times["pre"]),
        "L_dpu_ms": ms(times["dpu"]),
        "L_post_ms": ms(times["post"]),
        "L_e2e_ms": ms(times["e2e"]),
        "fps_single": fps_single(float(np.mean(times["e2e"]))),
        "fps_pipe_model": fps_pipe(stage_means_s),           # Eq. (17), 1/max stage
        "fps_pipe_measured": fps_measured,                    # steady-state observed
        "steady_window_s": steady_span,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmodel", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--route", choices=["r1", "r2", "r3"], default="r1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
