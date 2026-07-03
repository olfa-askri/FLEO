"""Auxiliary training losses that make orthogonalization a *regularizer*.

The whole premise of the fold-out (Section IV-B) is that Gram-Schmidt is useful
at training time and removable at inference.  For that to hold, the constraint it
imposes must be pushed into the *weights* during training.  Two auxiliary terms do
that:

* :func:`orthogonality_penalty` -- drives the K projected subspaces toward mutual
  orthogonality directly on the pre-orthogonalization projection ``p``, so the
  network learns to produce near-orthogonal features even before the GS map.
  This is what lets Route 1 survive removing GS.

* :func:`binding_loss` -- supervises the per-subspace binding head so each
  subspace specialises to one emotion.

These are added to the detector loss; they never touch the inference graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def orthogonality_penalty(p: torch.Tensor, k: int, eps: float = 1e-6) -> torch.Tensor:
    """Soft off-diagonal Gram penalty over the K flattened subspaces.

    ``p`` is the projection ``Conv1x1(X)`` of shape ``(B, K*d, H, W)``.  We form
    the normalized K x K Gram matrix per sample and penalise its off-diagonal
    mass, i.e. we ask the subspaces to be orthonormal *as produced*, so that the
    explicit GS map becomes redundant at inference.
    """
    b, c, h, w = p.shape
    v = p.reshape(b, k, -1)  # (B, K, D)
    v = v / (v.norm(dim=2, keepdim=True) + eps)
    gram = torch.bmm(v, v.transpose(1, 2))  # (B, K, K)
    eye = torch.eye(k, device=p.device, dtype=p.dtype).unsqueeze(0)
    off = gram - eye
    return off.pow(2).sum(dim=(1, 2)).mean()


def binding_loss(binding_logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that binds subspace i to emotion i.

    ``binding_logits`` has shape ``(B, K, num_emotions)``; the target for subspace
    ``i`` is class ``i`` (assuming ``K == num_emotions``).  When ``K`` differs the
    caller should pass a matching target instead; here we default to the diagonal
    binding described in the manuscript.
    """
    b, k, n = binding_logits.shape
    target = torch.arange(min(k, n), device=binding_logits.device)
    logits = binding_logits[:, : target.numel(), :].reshape(-1, n)
    tgt = target.repeat(b)
    return F.cross_entropy(logits, tgt)


def fleo_aux_loss(model, lambda_ortho: float = 0.01, lambda_bind: float = 0.01) -> torch.Tensor:
    """Sum the FLEO auxiliary losses over all FLEO sites for the current forward.

    Reads the transient ``_proj_out`` / ``_binding`` stashed on each FLEO module
    during a *training* forward and clears them afterwards.  Uses no hooks or
    closures, so nothing unpicklable is attached to the model.  Returns a scalar
    tensor (0 if there are no active FLEO sites, e.g. baseline or a folded graph).
    """
    from .fleo_block import FLEO

    sites = [m for m in model.modules() if isinstance(m, FLEO)]
    device = None
    for s in sites:
        device = s.proj.conv.weight.device
        break
    loss = torch.zeros((), device=device) if device is not None else torch.zeros(())
    for s in sites:
        if s._proj_out is not None:
            loss = loss + lambda_ortho * orthogonality_penalty(s._proj_out, s.k)
        if s._binding is not None:
            loss = loss + lambda_bind * binding_loss(s._binding)
        s._proj_out = None
        s._binding = None
    return loss
