"""Integrate FLEO into a YOLOv12-S detector via ultralytics.

We deliberately avoid editing ultralytics' ``parse_model`` (whose module sets are
local and version-fragile).  Instead we build the stock detector and then *wrap
the two neck layers that feed the Detect head at the P3 and P4 levels* with a
FLEO block, keeping each layer's index/from-list intact.  Because FLEO preserves
tensor shape and channel count (residual add), every downstream concat and the
save-list are unchanged -- the insertion is transparent to the runner.

Public helpers:
    build_fleo_yolo(cfg, nc, K, d, mode)  -> ultralytics.YOLO with FLEO necks
    set_route(model, mode)                -> switch all FLEO sites' route
    FLEODetectionModel                    -> DetectionModel that adds the FLEO aux loss
    register_fleo()                       -> best-effort YAML registration
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from .fleo_block import FLEO
from .losses import fleo_aux_loss


class FLEODetectionModel(DetectionModel):
    """DetectionModel that adds the FLEO auxiliary loss during training.

    A true module-level class so ultralytics checkpoints (which pickle the whole
    model) round-trip cleanly.  The lambdas are plain float attributes.
    """

    lambda_ortho = 0.01
    lambda_bind = 0.01

    def loss(self, batch, preds=None):
        total, items = super().loss(batch, preds)
        if self.training:
            total = total + fleo_aux_loss(self, self.lambda_ortho, self.lambda_bind)
        return total, items


class FLEOWrap(nn.Module):
    """Run the original neck layer, then a FLEO block, preserving layer metadata.

    ultralytics' forward loop reads ``.f`` (from-indices), ``.i`` (layer index),
    ``.type`` and ``.np`` off each layer; we mirror them so the wrapper is a drop-in.
    """

    def __init__(self, orig: nn.Module, fleo: FLEO) -> None:
        super().__init__()
        self.orig = orig
        self.fleo = fleo
        # Mirror ultralytics layer attributes.
        self.f = getattr(orig, "f", -1)
        self.i = getattr(orig, "i", -1)
        self.type = f"FLEOWrap({getattr(orig, 'type', orig.__class__.__name__)})"
        self.np = sum(x.numel() for x in self.parameters())

    def forward(self, x):
        return self.fleo(self.orig(x))


def _capture_out_channels(det_model, imgsz: int = 64):
    """Return a dict {layer_index: out_channels} via one dummy forward with hooks."""
    channels: dict[int, int] = {}
    handles = []

    def mk(idx):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t) and t.dim() == 4:
                channels[idx] = t.shape[1]
        return hook

    for idx, layer in enumerate(det_model.model):
        handles.append(layer.register_forward_hook(mk(idx)))

    was_training = det_model.training
    det_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        try:
            det_model(dummy)
        finally:
            for h in handles:
                h.remove()
            det_model.train(was_training)
    return channels


def _detect_p3_p4_indices(det_model):
    """The Detect head's first two input layers are the P3 and P4 neck outputs."""
    detect = det_model.model[-1]
    f = detect.f
    if not isinstance(f, (list, tuple)) or len(f) < 2:
        raise RuntimeError(f"Unexpected Detect.from indices: {f!r}")
    # Detect.f are absolute layer indices (negatives are relative to len-1).
    n = len(det_model.model)
    idxs = [(i if i >= 0 else n + i) for i in f]
    return idxs[0], idxs[1]  # P3, P4 (ascending stride order in YOLO necks)


def wrap_fleo(det, nc: int = 7, k: int = 7, d: int = 8, mode: str = "full", imgsz: int = 64):
    """Wrap the P3/P4 neck layers of an existing DetectionModel with FLEO (in place)."""
    channels = _capture_out_channels(det, imgsz=imgsz)
    p3, p4 = _detect_p3_p4_indices(det)
    for idx in (p3, p4):
        c1 = channels.get(idx)
        if c1 is None:
            raise RuntimeError(f"Could not infer channels for neck layer {idx}")
        if isinstance(det.model[idx], FLEOWrap):
            continue  # already wrapped
        orig = det.model[idx]
        fleo = FLEO(c1, k=k, d=d, num_emotions=nc, mode=mode)
        det.model[idx] = FLEOWrap(orig, fleo)
    det._fleo_sites = (p3, p4)
    return det


def build_fleo_detection_model(
    cfg: str = "yolo12s.yaml",
    nc: int = 7,
    k: int = 7,
    d: int = 8,
    mode: str = "full",
    imgsz: int = 64,
    lambda_ortho: float = 0.01,
    lambda_bind: float = 0.01,
):
    """Build a DetectionModel with the Detect head at ``nc`` and FLEO at P3/P4."""
    det = FLEODetectionModel(cfg, ch=3, nc=nc, verbose=False)
    det.lambda_ortho = lambda_ortho
    det.lambda_bind = lambda_bind
    det.names = {i: f"emotion_{i}" for i in range(nc)}
    wrap_fleo(det, nc=nc, k=k, d=d, mode=mode, imgsz=imgsz)
    return det


def build_fleo_yolo(
    cfg: str = "yolo12s.yaml",
    nc: int = 7,
    k: int = 7,
    d: int = 8,
    mode: str = "full",
    imgsz: int = 64,
):
    """Build a YOLOv12-S detector with FLEO wrapped at the P3 and P4 neck levels.

    Returns the ``ultralytics.YOLO`` wrapper; the underlying ``DetectionModel`` is
    ``model.model``.
    """
    from ultralytics import YOLO

    det = build_fleo_detection_model(cfg, nc=nc, k=k, d=d, mode=mode, imgsz=imgsz)
    yolo = YOLO(cfg)
    yolo.model = det
    return yolo


def register_torch_safe_globals():
    """Allowlist FLEO classes for ``torch.load(weights_only=True)``.

    Modern torch defaults to ``weights_only=True``; a FLEO checkpoint pickles the
    custom module classes, so they must be on the safe-globals allowlist or the
    load raises (or silently falls back).  Call this before ``YOLO(weights)`` on
    any FLEO checkpoint (export / evaluate / deltas).
    """
    import torch

    from .fleo_block import FLEO, ConvBNAct, BindingHead
    from .orthogonal import GramSchmidt, HouseholderOrtho

    try:
        torch.serialization.add_safe_globals(
            [FLEODetectionModel, FLEOWrap, FLEO, ConvBNAct, BindingHead,
             GramSchmidt, HouseholderOrtho]
        )
    except Exception:
        pass


def load_pretrained_into(det, weights):
    """Load matching (backbone/neck/head-by-name) weights from a .pt or model.

    Shape-mismatched tensors (e.g. an nc=80 COCO head vs our nc=7 head) and the
    FLEO-only modules are skipped by ultralytics' intersect_dicts, so this is a
    safe warm-start.  Returns the number of tensors actually copied.
    """
    if not weights:  # None or "" (scratch)
        return 0
    try:
        from ultralytics import YOLO

        src = YOLO(weights).model if isinstance(weights, str) else weights
        before = sum(v.numel() for v in det.state_dict().values())
        det.load(src)
        return before
    except Exception as e:  # never let a warm-start failure abort training
        print(f"[FLEO] pretrained warm-start skipped ({type(e).__name__}: {e})")
        return 0


def set_route(model, mode: str):
    """Switch every FLEO site to a deployment route ('full'|'folded'|'householder')."""
    det = getattr(model, "model", model)
    n = 0
    for m in det.modules():
        if isinstance(m, FLEO):
            if mode == "folded":
                m.fold_out()
            elif mode == "householder":
                m.fold_householder()
            else:
                m.set_full()
            n += 1
    return n


def make_fleo_trainer(k: int = 7, d: int = 8, mode: str = "full",
                      lambda_ortho: float = 0.01, lambda_bind: float = 0.01,
                      pretrained: str | None = None):
    """Return a DetectionTrainer subclass whose ``get_model`` yields a FLEO detector.

    Using a custom trainer is the robust way to keep FLEO through ``train()``:
    ultralytics rebuilds the model from the cfg inside ``setup_model``, so we make
    that rebuild produce the FLEO-wrapped model.  The auxiliary loss lives on the
    FLEODetectionModel class itself (picklable), so nothing extra is attached.

    ``pretrained`` (e.g. ``"yolo12s.pt"``) warm-starts the backbone/neck from COCO
    weights, which materially improves FER convergence; FLEO-only modules stay at
    their init.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FLEOTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            nc = self.data["nc"]
            det = build_fleo_detection_model(
                cfg or "yolo12s.yaml", nc=nc, k=k, d=d, mode=mode,
                lambda_ortho=lambda_ortho, lambda_bind=lambda_bind,
            )
            # ultralytics-supplied resume weights take precedence over COCO warm-start.
            load_pretrained_into(det, weights if weights is not None else pretrained)
            return det

    return FLEOTrainer


def register_fleo():
    """Best-effort: expose FLEO to ultralytics' YAML parser as a named module.

    Not required by :func:`build_fleo_yolo` (which wraps layers directly), but lets
    a hand-written YAML reference ``FLEO`` if desired.
    """
    try:
        import ultralytics.nn.tasks as tasks

        setattr(tasks, "FLEO", FLEO)
        return True
    except Exception:
        return False
