"""Voided vs non-voided encoder behaviour — ``docs/checks/03`` §7.

Tests V1 / V2 probe the *locality* of the MAISI encoder ``E`` on the
inference-time input distribution: the generator is conditioned on
``E(x_tilde)``, where ``x_tilde`` is the voided image. For a void mask
``m_v`` with latent-grid projection ``m_hat``:

- V1 (inside sensitivity):  ``S_inside  = ‖ (E(x) - E(x_tilde)) ⊙ m_hat ‖² / Σ m_hat``
- V2 (outside invariance):  ``S_outside = ‖ (E(x) - E(x_tilde)) ⊙ (1 - m_hat) ‖² / Σ (1 - m_hat)``

A useful encoder has a large ``S_inside`` (the latent separates void from
content) and a near-zero ``S_outside`` (voiding inside ``m_v`` does not perturb
the latent elsewhere). Both squared norms sum over the latent channels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

__all__ = [
    "VoidedTestResult",
    "compute_voided_tests_from_latents",
    "downsample_mask_to_latent",
]


@dataclass(frozen=True)
class VoidedTestResult:
    """Per-volume §7 results, aggregated over the J void masks (NaN-dropping)."""

    subject_id: str
    s_inside_mean: float
    s_inside_std: float
    s_outside_mean: float
    s_outside_std: float


def downsample_mask_to_latent(
    void_mask: np.ndarray,
    latent_shape: tuple[int, int, int],
) -> np.ndarray:
    """Project a voxel-space binary void mask onto the latent grid.

    Adaptive max-pooling marks a latent cell iff *any* voxel of its receptive
    block lies inside the void, and guarantees the output matches
    ``latent_shape`` regardless of the exact compression ratio.

    Parameters:
        void_mask: Binary mask ``(X, Y, Z)`` at the VAE envelope.
        latent_shape: Target latent spatial shape ``(Lz, Ly, Lx)``.

    Returns:
        Boolean array of shape ``latent_shape``.
    """
    t = torch.from_numpy(np.asarray(void_mask, dtype=np.float32))[None, None]
    pooled = F.adaptive_max_pool3d(t, output_size=tuple(latent_shape))
    return pooled[0, 0].numpy() > 0.5


def _as_cxyz(z: torch.Tensor | np.ndarray) -> np.ndarray:
    """Coerce a latent to ``(C, Lz, Ly, Lx)`` float64, dropping a batch axis."""
    if isinstance(z, torch.Tensor):
        arr = z.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        arr = np.asarray(z, dtype=np.float32)
    arr = arr.astype(np.float64)
    if arr.ndim == 5:  # (B, C, Lz, Ly, Lx) -> batch 0
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected a (C,Lz,Ly,Lx) latent, got {arr.ndim}-D")
    return arr


def compute_voided_tests_from_latents(
    *,
    subject_id: str,
    z_gt: torch.Tensor | np.ndarray,
    z_voided: torch.Tensor | np.ndarray,
    latent_void_masks: Sequence[np.ndarray],
) -> VoidedTestResult:
    """§7 V1/V2 statistics from a pair of encoded latents.

    Parameters:
        subject_id: Scan identifier.
        z_gt: Latent of the intact volume — ``(B,C,...)`` or ``(C,...)``.
        z_voided: Latent of the voided volume, same layout.
        latent_void_masks: J boolean masks on the latent grid (see
            :func:`downsample_mask_to_latent`).

    Returns:
        A frozen :class:`VoidedTestResult`.

    Raises:
        ValueError: If the two latents have different shapes.
    """
    diff = _as_cxyz(z_gt) - _as_cxyz(z_voided)  # (C, Lz, Ly, Lx)
    sq = diff * diff
    s_in: list[float] = []
    s_out: list[float] = []
    for m in latent_void_masks:
        mb = np.asarray(m).astype(bool)
        n_in = int(mb.sum())
        n_out = int((~mb).sum())
        s_in.append(float(sq[:, mb].sum()) / n_in if n_in else float("nan"))
        s_out.append(float(sq[:, ~mb].sum()) / n_out if n_out else float("nan"))
    s_in_a = np.array(s_in, dtype=np.float64)
    s_out_a = np.array(s_out, dtype=np.float64)

    def _mean(a: np.ndarray) -> float:
        keep = a[~np.isnan(a)]
        return float(keep.mean()) if keep.size else float("nan")

    def _std(a: np.ndarray) -> float:
        keep = a[~np.isnan(a)]
        return float(keep.std()) if keep.size else float("nan")

    return VoidedTestResult(
        subject_id=subject_id,
        s_inside_mean=_mean(s_in_a),
        s_inside_std=_std(s_in_a),
        s_outside_mean=_mean(s_out_a),
        s_outside_std=_std(s_out_a),
    )
