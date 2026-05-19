"""Mask-descriptor + KS audit used by pre-flight 01.

Four scalar descriptors per void mask (per ``docs/checks/01_augmentation_preflight.md`` §3.5):

- ``volume``: ``|m_v|`` in voxels.
- ``surface_to_volume``: count of boundary voxels (6-connectivity) divided by ``volume``.
- ``centroid_distance``: L2 distance between the centroid of ``m_v`` and the centroid of the
  brain mask, in voxels.
- ``max_diameter``: maximum diameter of the void along its PCA principal axis (in voxels).

Hard-fail rule (§4): KS p-value < 0.05 on ``volume`` or ``centroid_distance`` triggers
``ks_hard_fail`` and the engine aborts after writing ``decision.json``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

# The four descriptors in canonical order.
MASK_DESCRIPTORS: tuple[str, ...] = (
    "volume",
    "surface_to_volume",
    "centroid_distance",
    "max_diameter",
)

# Descriptors whose KS p-value < 0.05 trigger a hard-fail.
HARD_FAIL_DESCRIPTORS: tuple[str, ...] = ("volume", "centroid_distance")


def _boundary_count(mask: np.ndarray) -> int:
    """Count 6-connected boundary voxels of a binary mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return 0
    # A foreground voxel is a boundary voxel if at least one 6-neighbor is background.
    shifts = (
        np.pad(mask, ((1, 0), (0, 0), (0, 0)))[:-1] & mask,
        np.pad(mask, ((0, 1), (0, 0), (0, 0)))[1:] & mask,
        np.pad(mask, ((0, 0), (1, 0), (0, 0)))[:, :-1] & mask,
        np.pad(mask, ((0, 0), (0, 1), (0, 0)))[:, 1:] & mask,
        np.pad(mask, ((0, 0), (0, 0), (1, 0)))[:, :, :-1] & mask,
        np.pad(mask, ((0, 0), (0, 0), (0, 1)))[:, :, 1:] & mask,
    )
    has_all_6_neighbors = shifts[0] & shifts[1] & shifts[2] & shifts[3] & shifts[4] & shifts[5]
    boundary = mask & (~has_all_6_neighbors)
    return int(boundary.sum())


def _principal_axis_extent(mask: np.ndarray) -> float:
    """Maximum extent along the PCA principal axis of the mask coordinates (in voxels)."""
    mask = mask.astype(bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    if coords.shape[0] == 1:
        return 1.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # principal axis (largest eigenvalue)
    projections = centered @ axis
    return float(projections.max() - projections.min())


def compute_mask_descriptors(mask: np.ndarray, brain: np.ndarray) -> dict[str, float]:
    """Return the four scalar mask descriptors.

    Parameters:
        mask: Binary void mask ``(X, Y, Z)``.
        brain: Binary brain mask same shape (used for the centroid reference).

    Returns:
        ``{"volume", "surface_to_volume", "centroid_distance", "max_diameter"}`` as floats.
    """
    mask_bool = mask.astype(bool)
    brain_bool = brain.astype(bool)
    volume = int(mask_bool.sum())
    if volume == 0:
        return {
            "volume": 0.0,
            "surface_to_volume": 0.0,
            "centroid_distance": 0.0,
            "max_diameter": 0.0,
        }
    surface = _boundary_count(mask_bool)
    sv = float(surface) / float(volume)
    mask_centroid = np.array(ndi.center_of_mass(mask_bool))
    brain_centroid = np.array(ndi.center_of_mass(brain_bool)) if brain_bool.any() else mask_centroid
    centroid_distance = float(np.linalg.norm(mask_centroid - brain_centroid))
    max_diameter = _principal_axis_extent(mask_bool)
    return {
        "volume": float(volume),
        "surface_to_volume": sv,
        "centroid_distance": centroid_distance,
        "max_diameter": max_diameter,
    }


def ks_test(
    train_descriptors: dict[str, np.ndarray],
    val_descriptors: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-descriptor two-sample Kolmogorov-Smirnov test.

    Parameters:
        train_descriptors: Mapping from descriptor name to a 1-D array of values
            sampled at training time (across the chosen augmentation set).
        val_descriptors: Same shape, computed on the (frozen) validation-set masks.

    Returns:
        Mapping from descriptor name to p-value. Descriptors absent from either
        side default to NaN.
    """
    p_values: dict[str, float] = {}
    for name in MASK_DESCRIPTORS:
        train = train_descriptors.get(name)
        val = val_descriptors.get(name)
        if train is None or val is None or len(train) == 0 or len(val) == 0:
            p_values[name] = float("nan")
            continue
        stat = ks_2samp(train, val, alternative="two-sided", method="auto")
        p_values[name] = float(stat.pvalue)
    return p_values


def decide_hard_fail(p_values: dict[str, float]) -> bool:
    """True iff any hard-fail descriptor's p-value falls below 0.05.

    Per ``docs/checks/01_augmentation_preflight.md`` §4: ``volume`` and
    ``centroid_distance`` are the hard-fail descriptors.

    Parameters:
        p_values: Output of :func:`ks_test`.

    Returns:
        ``True`` ⇒ pre-flight 01 fails fast.
    """
    for name in HARD_FAIL_DESCRIPTORS:
        p = p_values.get(name, float("nan"))
        if np.isfinite(p) and p < 0.05:
            return True
    return False


__all__ = [
    "HARD_FAIL_DESCRIPTORS",
    "MASK_DESCRIPTORS",
    "compute_mask_descriptors",
    "decide_hard_fail",
    "ks_test",
]
