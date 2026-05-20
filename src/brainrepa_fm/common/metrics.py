"""Mask-aware reconstruction-quality metrics for ``[0, 1]``-valued volumes.

Public MSE / PSNR / SSIM helpers shared by pre-flight 03 (the MAISI VAE
reconstruction audit) and reusable by downstream evaluation routines. All
functions operate on 3-D ``(X, Y, Z)`` arrays whose intensities lie in
``[0, 1]``; masks are boolean-castable arrays of the same shape.

PSNR matches the convention of the private ``_psnr`` in
``brainrepa_fm.preflight.augmentation.vae_composability`` (``data_range=1.0``,
``inf`` on exact match, ``nan`` on an empty mask) so the two never disagree.

SSIM uses :func:`skimage.metrics.structural_similarity` with ``full=True`` to
obtain a per-voxel map; this is what allows a *region-restricted* SSIM (brain /
void / tumor) — a reduced scalar metric such as ``monai.metrics.SSIMMetric``
cannot be masked after the fact.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

__all__ = ["mse", "psnr", "ssim3d", "ssim3d_map"]


def _odd_window(win_size: int, shape: tuple[int, ...]) -> int:
    """Largest odd window ``<= win_size`` that also fits the smallest axis."""
    w = min(int(win_size), *shape)
    if w % 2 == 0:
        w -= 1
    return w


def mse(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean squared error, optionally restricted to ``mask``.

    Parameters:
        reference: Ground-truth volume ``(X, Y, Z)``.
        prediction: Reconstructed volume, same shape.
        mask: Optional boolean-castable region. When given but all-False, the
            result is ``nan``.

    Returns:
        MSE as a Python float (``nan`` for an empty mask).
    """
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if not m.any():
            return float("nan")
        diff = (np.asarray(prediction)[m] - np.asarray(reference)[m]).astype(np.float64)
    else:
        diff = (np.asarray(prediction) - np.asarray(reference)).astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    data_range: float = 1.0,
) -> float:
    """Peak signal-to-noise ratio in dB, optionally restricted to ``mask``.

    Parameters:
        reference: Ground-truth volume.
        prediction: Reconstructed volume.
        mask: Optional region restriction.
        data_range: Peak signal value (``1.0`` for ``[0, 1]`` volumes).

    Returns:
        PSNR in dB; ``inf`` on an exact match (MSE 0), ``nan`` for an empty mask.
    """
    val = mse(reference, prediction, mask)
    if np.isnan(val):
        return float("nan")
    if val <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10((float(data_range) ** 2) / val))


def ssim3d_map(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    win_size: int = 7,
    data_range: float = 1.0,
) -> np.ndarray:
    """Per-voxel 3-D SSIM map (uniform window).

    Computing the map once and reducing it over several masks is far cheaper
    than re-running SSIM per region; :func:`ssim3d` and
    :func:`brainrepa_fm.preflight.maisi_vae.reconstruction.compute_reconstruction_metrics`
    both build on this.

    Parameters:
        reference: Ground-truth volume ``(X, Y, Z)``.
        prediction: Reconstructed volume, same shape.
        win_size: Requested (odd) window edge; clamped to the smallest axis.
        data_range: Peak signal value.

    Returns:
        SSIM map of the same shape as the inputs (``float64``).

    Raises:
        ValueError: If the inputs are not 3-D or shapes differ.
    """
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if ref.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"ssim3d_map expects 3-D arrays, got {ref.ndim}-D / {pred.ndim}-D")
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {pred.shape}")
    w = _odd_window(win_size, ref.shape)
    if w < 3:
        return np.full(ref.shape, np.nan, dtype=np.float64)
    _, smap = structural_similarity(
        ref,
        pred,
        win_size=w,
        data_range=float(data_range),
        gaussian_weights=False,
        full=True,
    )
    return np.asarray(smap, dtype=np.float64)


def ssim3d(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    win_size: int = 7,
    data_range: float = 1.0,
) -> float:
    """3-D structural similarity, optionally restricted to ``mask``.

    Parameters:
        reference: Ground-truth volume.
        prediction: Reconstructed volume.
        mask: Optional region restriction; ``nan`` for an empty mask.
        win_size: Window edge (clamped odd to the smallest axis).
        data_range: Peak signal value.

    Returns:
        Mean SSIM over the volume (or the masked region).
    """
    smap = ssim3d_map(reference, prediction, win_size=win_size, data_range=data_range)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if not m.any():
            return float("nan")
        return float(np.mean(smap[m]))
    return float(np.mean(smap))
