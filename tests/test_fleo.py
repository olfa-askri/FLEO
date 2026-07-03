"""Unit tests for the FLEO core: orthogonality, the fold-out invariants, route
switching, ONNX op inventory, PS-GS parity, and checkpoint picklability."""

import numpy as np
import pytest
import torch

from fleo import FLEO, gram_schmidt, HouseholderOrtho
from fleo.orthogonal import GramSchmidt


def _gram(e, k):
    v = e.reshape(e.shape[0], k, -1)
    return torch.bmm(v, v.transpose(1, 2))


def test_gram_schmidt_orthonormal():
    torch.manual_seed(0)
    x = torch.randn(3, 7 * 8, 6, 6)
    e = gram_schmidt(x, k=7)
    g = _gram(e, 7)
    eye = torch.eye(7).expand_as(g)
    assert torch.allclose(g.diagonal(dim1=1, dim2=2), torch.ones(3, 7), atol=1e-4)
    off = (g - eye).abs()
    off = off - torch.diag_embed(off.diagonal(dim1=1, dim2=2))
    assert off.abs().max() < 1e-4


def test_gram_schmidt_differentiable():
    x = torch.randn(2, 7 * 4, 5, 5, requires_grad=True)
    e = GramSchmidt(7)(x)
    e.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_householder_orthogonal_and_export():
    torch.manual_seed(0)
    h = HouseholderOrtho(dim=56)
    assert h.orthogonality_error() < 1e-2
    conv = h.export_conv1x1()
    x = torch.randn(2, 56, 4, 4)
    assert torch.allclose(h(x), conv(x), atol=1e-5)


def test_fleo_shape_and_foldout_preserves_weights():
    f = FLEO(64, k=7, d=8).eval()
    x = torch.randn(2, 64, 8, 8)
    y_full = f(x)
    assert y_full.shape == x.shape
    w = f.proj.conv.weight.clone()
    f.fold_out()
    y_fold = f(x)
    assert f.mode == "folded"
    assert torch.equal(w, f.proj.conv.weight)   # weights untouched by the rewrite
    assert y_fold.shape == x.shape
    # folded != full in general (GS actually changed the activations)
    assert not torch.allclose(y_full, y_fold)


def test_fleo_no_transient_state_in_eval():
    f = FLEO(32).eval()
    f(torch.randn(1, 32, 4, 4))
    assert f._proj_out is None and f._binding is None  # picklable at export


def test_ps_gs_matches_torch():
    from deploy.app.gs_ps import gram_schmidt_np

    torch.manual_seed(1)
    x = torch.randn(2, 7 * 8, 4, 4)
    e_t = gram_schmidt(x, 7).numpy()
    e_n = gram_schmidt_np(x.numpy(), 7)
    assert np.abs(e_t - e_n).max() < 1e-4


def test_metrics_constants():
    from deploy import metrics as M

    assert abs(M.peak_tops() - 1.229) < 1e-3
    assert abs(M.ridge_point() - 64.0) < 1e-6
    assert abs(M.fmax_hz(3.333, 0.0) - 300e6) < 1e5


@pytest.mark.slow
def test_detection_model_routes_and_pickle(tmp_path):
    from fleo.yolo_integration import build_fleo_detection_model, set_route, FLEODetectionModel

    det = build_fleo_detection_model("yolo12s.yaml", nc=7, k=7, d=8, imgsz=64)
    det.eval()
    x = torch.zeros(1, 3, 96, 96)
    for mode in ("full", "folded", "householder", "full"):
        n = set_route(det, mode)
        assert n == 2
        with torch.no_grad():
            out = det(x)
        assert out is not None

    # The whole model must pickle (ultralytics saves it at every epoch).
    p = tmp_path / "ckpt.pt"
    torch.save(det, p)
    torch.serialization.add_safe_globals([FLEODetectionModel])
    det2 = torch.load(p, weights_only=False)
    assert sum(p.numel() for p in det2.parameters()) == sum(p.numel() for p in det.parameters())
