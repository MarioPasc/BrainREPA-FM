"""Round-trip reconstruction metrics for the MAISI VAE audit (pre-flight 03).

Implements ``docs/checks/03_maisi_vae_audit.md`` §2.1 / §3.2: per-volume
MSE / PSNR / SSIM of the round-trip ``r(x) = D(E(x))`` in four regions — the
full envelope, the brain mask, J sampled void masks, and the tumor mask.
:func:`compute_voided_roundtrip_metrics` covers Caveat 2 — the round-trip of
the *voided* volume ``r(x_tilde) = D(E(x_tilde))``, scoring whether the input
hole degrades the decoder's rendering of the still-visible tissue.

Both functions are pure (no VAE, no GPU): they consume already-decoded
reconstructions, which keeps the metric maths unit-testable without a
checkpoint. The routine engine owns the encode/decode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from brainrepa_fm.common.metrics import mse, psnr, ssim3d_map

__all__ = [
    "ReconstructionMetrics",
    "VoidedRoundtripMetrics",
    "compute_reconstruction_metrics",
    "compute_voided_roundtrip_metrics",
]

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


def _delta(better: float, worse: float) -> float:
    """``better - worse`` when both are finite, else ``nan`` (avoids ``inf - inf``)."""
    if not (np.isfinite(better) and np.isfinite(worse)):
        return float("nan")
    return float(better - worse)


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


@dataclass(frozen=True)
class VoidedRoundtripMetrics:
    """Voided-input round-trip fidelity for one volume (``docs/checks/03`` Caveat 2).

    At inference the generator is conditioned on ``E(x_tilde)`` where ``x_tilde``
    is the voided volume; this measures whether the zero hole degrades the
    decoder's rendering of the *still-visible* tissue. The ``visible`` region is
    ``brain ∩ ¬void`` — voxels present in both ``x`` and ``x_tilde`` (where the
    two are equal by construction). ``*_visible_voided`` scores
    ``D(E(x_tilde)))`` there against the true signal ``x``; ``*_visible_unvoided``
    scores ``D(E(x))`` on the same region, so ``delta_psnr_visible_db``
    (un-voided minus voided) isolates the fidelity lost purely to the input
    hole. ``*_full_voided`` scores ``D(E(x_tilde)))`` against ``x_tilde`` over
    the whole envelope. ``delta_*`` are ``nan`` when either side is non-finite.
    """

    subject_id: str
    mse_visible_voided: float
    psnr_visible_voided: float
    ssim_visible_voided: float
    mse_visible_unvoided: float
    psnr_visible_unvoided: float
    ssim_visible_unvoided: float
    delta_psnr_visible_db: float
    delta_ssim_visible: float
    mse_full_voided: float
    psnr_full_voided: float
    ssim_full_voided: float


def compute_voided_roundtrip_metrics(
    *,
    subject_id: str,
    gt: np.ndarray,
    voided: np.ndarray,
    recon_unvoided: np.ndarray,
    recon_voided: np.ndarray,
    brain_mask: np.ndarray,
    void_mask: np.ndarray,
) -> VoidedRoundtripMetrics:
    """Voided-input round-trip fidelity for one volume; arrays at the VAE envelope.

    Parameters:
        subject_id: Scan identifier persisted into the result.
        gt: Intact volume ``x`` ``(X, Y, Z)`` in ``[0, 1]``.
        voided: Voided volume ``x_tilde`` (zeros inside ``void_mask``), same shape.
        recon_unvoided: Round-trip ``D(E(x))``, same shape.
        recon_voided: Round-trip ``D(E(x_tilde))``, same shape.
        brain_mask: Binary brain mask, same shape.
        void_mask: The binary void mask that produced ``voided``, same shape.

    Returns:
        A frozen :class:`VoidedRoundtripMetrics`.

    Raises:
        ValueError: If any input shape differs from ``gt``.
    """
    gt = np.asarray(gt, dtype=np.float64)
    voided = np.asarray(voided, dtype=np.float64)
    recon_unvoided = np.asarray(recon_unvoided, dtype=np.float64)
    recon_voided = np.asarray(recon_voided, dtype=np.float64)
    for name, arr in (
        ("voided", voided),
        ("recon_unvoided", recon_unvoided),
        ("recon_voided", recon_voided),
    ):
        if arr.shape != gt.shape:
            raise ValueError(f"{name} shape {arr.shape} != gt shape {gt.shape}")

    brain_b = np.asarray(brain_mask).astype(bool)
    void_b = np.asarray(void_mask).astype(bool)
    # Visible tissue: in the brain and outside the void — present in both x and x_tilde.
    visible = brain_b & ~void_b

    # Score both round-trips against the true signal x on the visible region, so
    # the delta is apples-to-apples; the full-envelope number scores x_tilde as-is.
    smap_voided = ssim3d_map(gt, recon_voided)
    smap_unvoided = ssim3d_map(gt, recon_unvoided)
    smap_full = ssim3d_map(voided, recon_voided)

    psnr_vv = psnr(gt, recon_voided, visible)
    psnr_vu = psnr(gt, recon_unvoided, visible)
    ssim_vv = _ssim_region(smap_voided, visible)
    ssim_vu = _ssim_region(smap_unvoided, visible)

    return VoidedRoundtripMetrics(
        subject_id=subject_id,
        mse_visible_voided=mse(gt, recon_voided, visible),
        psnr_visible_voided=psnr_vv,
        ssim_visible_voided=ssim_vv,
        mse_visible_unvoided=mse(gt, recon_unvoided, visible),
        psnr_visible_unvoided=psnr_vu,
        ssim_visible_unvoided=ssim_vu,
        delta_psnr_visible_db=_delta(psnr_vu, psnr_vv),
        delta_ssim_visible=_delta(ssim_vu, ssim_vv),
        mse_full_voided=mse(voided, recon_voided),
        psnr_full_voided=psnr(voided, recon_voided),
        ssim_full_voided=_ssim_region(smap_full, None),
    )
