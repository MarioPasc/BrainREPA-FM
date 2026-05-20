"""Round-trip reconstruction metrics for the MAISI VAE audit (pre-flight 03).

Implements ``docs/checks/03_maisi_vae_audit.md`` §2.1 / §3.2: per-volume
MSE / PSNR / SSIM of the round-trip ``r(x) = D(E(x))`` in four regions — the
full envelope, the brain mask, J sampled void masks, and the tumor mask.

:func:`compute_reconstruction_metrics` is pure (no VAE, no GPU): it consumes an
already-decoded reconstruction and returns a frozen :class:`ReconstructionMetrics`.
The routine engine owns the encode/decode, which keeps the metric maths
unit-testable without a checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from brainrepa_fm.common.metrics import mse, psnr, ssim3d_map

__all__ = ["ReconstructionMetrics", "compute_reconstruction_metrics"]

# Percentile (over the J void masks) for the per-volume worst case.
_WORST_PCT: float = 10.0


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Per-volume round-trip metrics (``docs/checks/03`` §3.2).

    All metrics assume intensities in ``[0, 1]``. The ``void`` family
    aggregates over the J sampled void masks: ``*_mean`` is the mean over j;
    ``psnr_void_worst`` is the 10th-percentile PSNR and ``mse_void_worst`` the
    90th-percentile MSE — both name the same pessimistic tail (low PSNR / high
    error). NaN values (empty masks) are dropped before aggregation.
    """

    subject_id: str
    mse_full: float
    psnr_full: float
    ssim_full: float
    mse_brain: float
    psnr_brain: float
    ssim_brain: float
    mse_void_mean: float
    mse_void_worst: float
    psnr_void_mean: float
    psnr_void_worst: float
    ssim_void_mean: float
    mse_tumor: float
    psnr_tumor: float
    ssim_tumor: float


def _ssim_region(smap: np.ndarray, mask: np.ndarray | None) -> float:
    """Reduce a precomputed per-voxel SSIM map over a region."""
    if mask is None:
        return float(np.mean(smap))
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return float("nan")
    return float(np.mean(smap[m]))


def _agg(reducer: Callable[[np.ndarray], float], values: np.ndarray) -> float:
    """Apply ``reducer`` to the non-NaN entries (``inf`` is kept — a real best case)."""
    keep = values[~np.isnan(values)]
    return float(reducer(keep)) if keep.size else float("nan")


def compute_reconstruction_metrics(
    *,
    subject_id: str,
    gt: np.ndarray,
    recon: np.ndarray,
    brain_mask: np.ndarray,
    tumor_mask: np.ndarray,
    void_masks: Sequence[np.ndarray],
) -> ReconstructionMetrics:
    """Round-trip metrics for one volume; every array at the VAE envelope.

    Parameters:
        subject_id: Scan identifier persisted into the result.
        gt: Intact ground-truth volume ``(X, Y, Z)`` in ``[0, 1]``.
        recon: Round-trip reconstruction ``D(E(gt))``, same shape.
        brain_mask: Binary brain mask, same shape.
        tumor_mask: Binary tumor mask, same shape.
        void_masks: J binary void masks, each the same shape.

    Returns:
        A frozen :class:`ReconstructionMetrics`.

    Raises:
        ValueError: If ``gt`` and ``recon`` shapes differ.
    """
    gt = np.asarray(gt, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if gt.shape != recon.shape:
        raise ValueError(f"gt/recon shape mismatch: {gt.shape} vs {recon.shape}")

    # The SSIM map is the only expensive term — compute once, reduce per region.
    smap = ssim3d_map(gt, recon)
    brain_b = np.asarray(brain_mask).astype(bool)
    tumor_b = np.asarray(tumor_mask).astype(bool)

    psnr_void = np.array([psnr(gt, recon, vm) for vm in void_masks], dtype=np.float64)
    mse_void = np.array([mse(gt, recon, vm) for vm in void_masks], dtype=np.float64)
    ssim_void = np.array([_ssim_region(smap, vm) for vm in void_masks], dtype=np.float64)

    return ReconstructionMetrics(
        subject_id=subject_id,
        mse_full=mse(gt, recon),
        psnr_full=psnr(gt, recon),
        ssim_full=_ssim_region(smap, None),
        mse_brain=mse(gt, recon, brain_b),
        psnr_brain=psnr(gt, recon, brain_b),
        ssim_brain=_ssim_region(smap, brain_b),
        mse_void_mean=_agg(np.mean, mse_void),
        mse_void_worst=_agg(
            lambda a: np.percentile(a, 100.0 - _WORST_PCT, method="lower"), mse_void
        ),
        psnr_void_mean=_agg(np.mean, psnr_void),
        psnr_void_worst=_agg(lambda a: np.percentile(a, _WORST_PCT, method="lower"), psnr_void),
        ssim_void_mean=_agg(np.mean, ssim_void),
        mse_tumor=mse(gt, recon, tumor_b),
        psnr_tumor=psnr(gt, recon, tumor_b),
        ssim_tumor=_ssim_region(smap, tumor_b),
    )
