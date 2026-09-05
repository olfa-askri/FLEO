"""Grad-CAM face heatmaps for the trained YOLOv12-S + FLEO emotion detector.

This produces the qualitative "where does the model look?" figure asked for in
review Comment 2.9 (heatmap figure): a **real** class-activation map computed
from your trained weights (`runs/fleo/fleo_seed0/weights/best.pt`) and overlaid
on face images. Nothing here is synthetic -- the map comes from the checkpoint's
own activations and gradients, so it is only meaningful when you point it at a
real checkpoint and real faces.

Why it is not a textbook classification Grad-CAM
------------------------------------------------
FER is cast as *detection* here (one full-frame emotion box per image). So the
CAM is taken on the **FLEO neck feature maps at P3/P4** -- exactly the tensors
that feed the Detect head -- and the scalar we backpropagate is the Detect
head's confidence for the target emotion (per-class max over anchors, the same
top-1 rule used in `scripts/evaluate.py`). That makes the heatmap show what the
*FLEO-augmented neck* attends to, not just the raw backbone.

    method = "gradcam"  (default): gradient-weighted CAM (Selvaraju et al.).
    method = "eigencam"          : gradient-free CAM (first principal component
                                   of the feature map; robust when the head's
                                   grad path is awkward). Used automatically as a
                                   per-layer fallback if a gradient never lands.

Usage (Kaggle, after training so best.pt + datasets/ exist)
-----------------------------------------------------------
    # One face per emotion, sampled from the prepared FER2013 val split:
    python -m scripts.gradcam \
        --weights runs/fleo/fleo_seed0/weights/best.pt \
        --data datasets/fer2013/data.yaml \
        --imgsz 160 --out results/gradcam

    # Or point it at specific face images (any size / color or grayscale):
    python -m scripts.gradcam --weights runs/fleo/fleo_seed0/weights/best.pt \
        --images face1.jpg face2.png --out results/gradcam

Outputs (under --out, default results/gradcam):
    gradcam_faces.png            -- grid: input | Grad-CAM overlay, one row/face
    gradcam_<i>_<emotion>.png    -- each overlay on its own
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Make `python scripts/gradcam.py` work as well as `python -m scripts.gradcam`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.emotions import CANON  # noqa: E402  (canonical 7-emotion order)


# --------------------------------------------------------------------------- io
def letterbox(im: Image.Image, imgsz: int = 160, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square, exactly like
    `scripts.evaluate._letterbox` (so the model sees what it saw at eval time).

    Returns (chw_input 1x3xHxW float32 in [0,1], display HxWx3 float in [0,1]).
    """
    im = im.convert("RGB")
    w, h = im.size
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im_r = im.resize((nw, nh))
    canvas = Image.new("RGB", (imgsz, imgsz), color)
    canvas.paste(im_r, ((imgsz - nw) // 2, (imgsz - nh) // 2))
    disp = np.asarray(canvas, dtype=np.float32) / 255.0     # HxWx3 [0,1]
    chw = disp.transpose(2, 0, 1)[None].copy()              # 1x3xHxW
    return chw, disp


def _cmap(name="jet"):
    import matplotlib

    try:
        return matplotlib.colormaps[name]                  # mpl >= 3.5
    except Exception:                                       # pragma: no cover
        import matplotlib.cm as cm

        return cm.get_cmap(name)


def overlay(disp: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """Blend a [0,1] heatmap over the [0,1] display image."""
    heat = _cmap("jet")(cam)[..., :3]                       # HxWx3
    return np.clip((1.0 - alpha) * disp + alpha * heat, 0.0, 1.0)


# ----------------------------------------------------------------------- model
def load_detector(weights: str, route: str, device: str):
    """Load an ultralytics FLEO checkpoint into an eval-ready DetectionModel."""
    import torch
    from ultralytics import YOLO
    from fleo.yolo_integration import register_torch_safe_globals, set_route

    register_torch_safe_globals()                          # unpickle FLEO classes
    det = YOLO(weights).model.to(device).eval()
    set_route(det, route)                                  # 'full' = trained graph
    nc = det.model[-1].nc
    return det, nc


# ------------------------------------------------------------------- Grad-CAM
class NeckCAM:
    """Grad-CAM / Eigen-CAM on the FLEO neck sites (or the P3/P4 Detect inputs
    for a baseline checkpoint that has no FLEO block)."""

    def __init__(self, det, method: str = "gradcam"):
        import torch

        self.torch = torch
        self.det = det
        self.method = method
        self.acts: dict = {}
        self.grads: dict = {}
        self.handles = []
        for name, module in self._find_targets().items():
            self.handles.append(module.register_forward_hook(self._fwd(name)))

    def _find_targets(self):
        """Prefer the FLEO-wrapped neck layers; fall back to the two layers that
        feed the Detect head."""
        layers = list(self.det.model)
        wraps = {f"site{j}": m for j, m in enumerate(layers)
                 if m.__class__.__name__ == "FLEOWrap"}
        if wraps:
            return wraps
        detect = layers[-1]
        f = detect.f
        f = f if isinstance(f, (list, tuple)) else [f]
        n = len(layers)
        idxs = [(i if i >= 0 else n + i) for i in f][:2]
        return {f"neck{idx}": layers[idx] for idx in idxs}

    def _fwd(self, name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            self.acts[name] = t
            if self.method == "gradcam" and t.requires_grad:
                t.register_hook(lambda g, n=name: self.grads.__setitem__(n, g))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    # -- head class scores (grad-carrying) --------------------------------
    def _cls_scores(self, preds, nc):
        if preds.dim() == 3 and preds.shape[1] == 4 + nc:      # (1, 4+nc, N)
            return preds[0, 4:4 + nc, :]
        if preds.dim() == 3 and preds.shape[2] == 4 + nc:      # (1, N, 4+nc)
            return preds[0, :, 4:4 + nc].transpose(0, 1)
        raise RuntimeError(f"unexpected Detect output {tuple(preds.shape)} (nc={nc})")

    # -- CAM math ---------------------------------------------------------
    def _eigen(self, act):
        """Eigen-CAM: project the feature map onto its first principal component."""
        torch = self.torch
        A = act[0].detach().float()                            # (C, h, w)
        C, h, w = A.shape
        M = A.reshape(C, h * w)
        M = M - M.mean(dim=1, keepdim=True)
        try:
            U, _, _ = torch.linalg.svd(M, full_matrices=False)
            proj = (U[:, 0:1].transpose(0, 1) @ M).reshape(h, w)
        except Exception:
            proj = M.abs().mean(0).reshape(h, w)               # safe last resort
        if proj.mean() < 0:                                    # fix sign ambiguity
            proj = -proj
        return proj.clamp(min=0)[None, None]                   # (1,1,h,w)

    def _gradcam_map(self, act, grad):
        w = grad.mean(dim=(2, 3), keepdim=True)                # (1, C, 1, 1)
        cam = (w * act).sum(1, keepdim=True).clamp(min=0)      # (1, 1, h, w)
        return cam.detach()

    def _merge(self, cams, size, layer):
        """Upsample each site CAM to `size`, normalize, and average the chosen
        subset. Sites are ordered finest->coarsest (P3 then P4)."""
        torch = self.torch
        import torch.nn.functional as F

        items = sorted(cams.items(), key=lambda kv: kv[1].shape[-1] * kv[1].shape[-2],
                       reverse=True)                           # finest first
        if layer == "p3":
            items = items[:1]
        elif layer == "p4":
            items = items[-1:]
        ups = []
        for _, c in items:
            c = F.interpolate(c.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
            c = c - c.min()
            c = c / (c.max() + 1e-8)
            ups.append(c)
        m = torch.stack(ups).mean(0)
        m = m - m.min()
        m = m / (m.max() + 1e-8)
        return m.cpu().numpy()

    def __call__(self, x, nc, class_idx=None, layer="both"):
        torch = self.torch
        self.acts.clear()
        self.grads.clear()
        size = tuple(x.shape[-2:])

        if self.method == "eigencam":
            with torch.no_grad():
                self.det(x)
            cams = {n: self._eigen(a) for n, a in self.acts.items()}
            return self._merge(cams, size, layer), class_idx, None

        # Grad-CAM: forward WITH grad, score the target emotion, backprop.
        out = self.det(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out
        cls = self._cls_scores(preds, nc)                      # (nc, N), in [0,1]
        if class_idx is None:
            class_idx = int(cls.amax(dim=1).argmax().item())   # top-1 emotion
        score = cls[class_idx].amax()
        self.det.zero_grad(set_to_none=True)
        score.backward()

        cams = {}
        for n, a in self.acts.items():
            g = self.grads.get(n)
            cams[n] = self._gradcam_map(a, g) if g is not None else self._eigen(a)
        return self._merge(cams, size, layer), class_idx, float(score.detach())


# --------------------------------------------------------------------- samples
def _resolve_names(data_yaml):
    if not data_yaml:
        return {i: n for i, n in enumerate(CANON)}
    import yaml

    names = yaml.safe_load(open(data_yaml))["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def collect_samples(args):
    """Return a list of (image_path, true_class_id_or_None)."""
    if args.images:
        return [(p, None) for p in args.images]

    from scripts.evaluate import _list_val

    imgs, gts, _ = _list_val(args.data)
    if not imgs:
        raise SystemExit(f"No labelled val images found via {args.data}")
    # One image per class first (nice per-emotion figure), then fill to --num.
    chosen, seen = [], set()
    for p, g in zip(imgs, gts):
        if g not in seen:
            seen.add(g)
            chosen.append((str(p), int(g)))
    for p, g in zip(imgs, gts):                                 # top up if needed
        if len(chosen) >= args.num:
            break
        chosen.append((str(p), int(g)))
    return chosen[:args.num]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/fleo/fleo_seed0/weights/best.pt",
                    help="ultralytics FLEO .pt checkpoint")
    ap.add_argument("--data", default="datasets/fer2013/data.yaml",
                    help="data.yaml to sample faces from (ignored if --images)")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit face image paths (overrides --data sampling)")
    ap.add_argument("--num", type=int, default=7, help="how many faces to show")
    ap.add_argument("--imgsz", type=int, default=160, help="square input size")
    ap.add_argument("--route", choices=["full", "folded", "householder"],
                    default="full", help="'full' = the trained Gram-Schmidt graph")
    ap.add_argument("--method", choices=["gradcam", "eigencam"], default="gradcam")
    ap.add_argument("--layer", choices=["p3", "p4", "both"], default="both",
                    help="which FLEO neck site(s) to visualise")
    ap.add_argument("--orient", choices=["vertical", "horizontal"], default="vertical",
                    help="figure layout: 'vertical' (rows=faces, input|overlay) or "
                         "'horizontal' (2 rows: faces on top, Grad-CAM below)")
    ap.add_argument("--target-class", default=None,
                    help="force the scored emotion (id or name); default = top-1")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay opacity")
    ap.add_argument("--out", default="results/gradcam")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    det, nc = load_detector(args.weights, args.route, device)
    for p in det.parameters():                                # grad must flow
        p.requires_grad_(True)
    names = _resolve_names(None if args.images else args.data)

    forced = None
    if args.target_class is not None:
        t = args.target_class
        if str(t).isdigit():
            forced = int(t)
        else:
            lut = {v.lower(): k for k, v in names.items()}
            lut.update({n.lower(): i for i, n in enumerate(CANON)})
            if t.lower() not in lut:
                raise SystemExit(f"--target-class '{t}' not in {list(names.values())}")
            forced = lut[t.lower()]

    cam_engine = NeckCAM(det, method=args.method)
    samples = collect_samples(args)
    print(f"[gradcam] {len(samples)} face(s) | device={device} | imgsz={args.imgsz} "
          f"| method={args.method} | layer={args.layer} | route={args.route}")

    results = []
    for i, (path, true_id) in enumerate(samples):
        x_np, disp = letterbox(Image.open(path), args.imgsz)
        x = torch.from_numpy(x_np).to(device)
        cam, pred_id, conf = cam_engine(x, nc, class_idx=forced, layer=args.layer)

        pred_name = names.get(pred_id, str(pred_id))
        true_name = names.get(true_id, None) if true_id is not None else None
        results.append((disp, cam, true_name, pred_name, conf, os.path.basename(path)))

        # individual overlay
        over = overlay(disp, cam, args.alpha)
        tag = (true_name or f"img{i}")
        ind = os.path.join(args.out, f"gradcam_{i}_{tag}.png")
        plt.imsave(ind, over)
        c = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"  [{i}] {os.path.basename(path):<28} true={true_name}  "
              f"pred={pred_name} ({c})  -> {ind}")

    # combined figure (layout chosen by --orient)
    n = len(results)
    title = f"FLEO neck {args.method} ({args.layer.upper()}) - where the detector attends"
    if args.orient == "horizontal":
        # 2 rows x n cols: faces on top, Grad-CAM overlays below (paper style).
        fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.9), squeeze=False)
        for i, (disp, cam, true_name, pred_name, conf, fname) in enumerate(results):
            c = f" ({conf:.2f})" if conf is not None else ""
            axes[0, i].imshow(disp)
            axes[0, i].set_title((f"{true_name}\n" if true_name else "") + f"{pred_name}{c}",
                                 fontsize=9)
            axes[1, i].imshow(overlay(disp, cam, args.alpha))
            for ax in (axes[0, i], axes[1, i]):
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
        axes[0, 0].set_ylabel("input", fontsize=10)
        axes[1, 0].set_ylabel(args.method, fontsize=10)
    else:
        # n rows x 2 cols: input | overlay, one face per row.
        fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.5 * n), squeeze=False)
        for r, (disp, cam, true_name, pred_name, conf, fname) in enumerate(results):
            axes[r, 0].imshow(disp)
            axes[r, 0].set_title(f"input: {true_name or fname}", fontsize=9)
            axes[r, 0].axis("off")
            axes[r, 1].imshow(overlay(disp, cam, args.alpha))
            c = f" ({conf:.2f})" if conf is not None else ""
            axes[r, 1].set_title(f"{args.method} -> {pred_name}{c}", fontsize=9)
            axes[r, 1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    grid = os.path.join(args.out, "gradcam_faces.png")
    fig.savefig(grid, dpi=150)
    plt.close(fig)
    cam_engine.remove()
    print(f"[gradcam] wrote {grid}")


if __name__ == "__main__":
    main()
