"""FLEO: a DPU-native FPGA deployment methodology for an orthogonalization-
regularized facial-emotion detector.

Public API:
    FLEO, GramSchmidt, HouseholderOrtho, gram_schmidt
    fleo_aux_loss, orthogonality_penalty, binding_loss
    register_fleo   -- inject FLEO into ultralytics so custom YAMLs resolve it.
"""

from .orthogonal import GramSchmidt, HouseholderOrtho, gram_schmidt
from .fleo_block import FLEO, MODES
from .losses import fleo_aux_loss, orthogonality_penalty, binding_loss

__all__ = [
    "FLEO",
    "MODES",
    "GramSchmidt",
    "HouseholderOrtho",
    "gram_schmidt",
    "fleo_aux_loss",
    "orthogonality_penalty",
    "binding_loss",
    "register_fleo",
]


def register_fleo():
    """Register FLEO with ultralytics' model parser. Import lazily so the core
    modules do not require ultralytics to be installed."""
    from .yolo_integration import register_fleo as _r

    return _r()
