"""Visualization helpers for pre-flight 01.

Two figure families per ``docs/checks/01_augmentation_preflight.md`` §3.3 & §3.5:

- 16 QC grids: 8 transforms × 2 subjects (S★, S†), each a 5-row × 3-column
  layout of mid-axial / mid-sagittal / mid-coronal slices. Rows: original,
  transformed, reconstructed, residual, intra-brain histogram.
- 4 KS CDF figures: one per mask descriptor, empirical CDFs for train (the
  chosen augmentation set's pooled samples) vs validation (frozen on-disk
  masks), with the p-value annotated.

All figures are written as PNG at 200 dpi.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _slice_axial(vol: np.ndarray) -> np.ndarray:
    return vol[:, :, vol.shape[2] // 2]


def _slice_sagittal(vol: np.ndarray) -> np.ndarray:
    return vol[vol.shape[0] // 2, :, :]


def _slice_coronal(vol: np.ndarray) -> np.ndarray:
    return vol[:, vol.shape[1] // 2, :]


def render_qc_grid(
    *,
    t1_baseline: np.ndarray,
    t1_aug: np.ndarray,
    decoded_aug: np.ndarray,
    void_aug: np.ndarray,
    brain: np.ndarray,
    transform_id: str,
    subject_label: str,
    out_path: Path,
) -> Path:
    """Render the 5×3 QC grid for one (transform, subject).

    Parameters:
        t1_baseline: Original voided T1 ``(X, Y, Z)`` in [0, 1].
        t1_aug: Augmented voided T1.
        decoded_aug: ``D(E(T(x)))``, same shape.
        void_aug: Binary void mask of the augmented pass.
        brain: Brain mask (used for histogram restriction).
        transform_id: e.g. ``"C.2"``.
        subject_label: ``"S★"`` or ``"S†"``.
        out_path: PNG destination (parent must exist).

    Returns:
        ``out_path``.
    """
    fig, axes = plt.subplots(5, 3, figsize=(9, 14))
    fig.suptitle(f"{transform_id} — {subject_label}", fontsize=14)

    slicers = (_slice_axial, _slice_sagittal, _slice_coronal)
    titles = ("axial", "sagittal", "coronal")
    diff = np.abs(decoded_aug - t1_aug)
    vmax_diff = float(np.percentile(diff[brain.astype(bool)], 99)) if brain.any() else 1.0
    if vmax_diff <= 0:
        vmax_diff = 1.0

    for col, (sl, title) in enumerate(zip(slicers, titles)):
        axes[0, col].imshow(sl(t1_baseline).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"baseline x ({title})")
        axes[1, col].imshow(sl(t1_aug).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"T(x) ({title})")
        axes[2, col].imshow(sl(decoded_aug).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[2, col].set_title(f"D(E(T(x))) ({title})")
        axes[3, col].imshow(sl(diff).T, cmap="hot", origin="lower", vmin=0, vmax=vmax_diff)
        # Overlay void edge
        ax3 = axes[3, col]
        void_slice = sl(void_aug)
        if void_slice.any():
            ax3.contour(void_slice.T, levels=[0.5], colors="cyan", linewidths=0.8)
        ax3.set_title(f"|D(E(T(x))) − T(x)| ({title})")
        ax3.set_facecolor("black")
        for r in range(4):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # Row 5: histogram inside brain.
    brain_bool = brain.astype(bool)
    if brain_bool.any():
        for col, (vol, label) in enumerate(
            zip(
                (t1_baseline, t1_aug, decoded_aug),
                ("baseline", "T(x)", "D(E(T(x)))"),
            )
        ):
            axes[4, col].hist(vol[brain_bool].ravel(), bins=64, range=(0, 1), color="steelblue")
            axes[4, col].set_title(f"intra-brain hist — {label}")
            axes[4, col].set_yscale("log")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_ks_cdf(
    train_values: np.ndarray,
    val_values: np.ndarray,
    *,
    descriptor: str,
    p_value: float,
    out_path: Path,
) -> Path:
    """Render a single-descriptor train-vs-val empirical CDF figure.

    Parameters:
        train_values: 1-D array of training-distribution descriptor values.
        val_values: 1-D array of validation-distribution descriptor values.
        descriptor: e.g. ``"volume"``.
        p_value: KS p-value, written into the title.
        out_path: PNG destination.

    Returns:
        ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for arr, label, color in (
        (np.sort(train_values), "train", "tab:blue"),
        (np.sort(val_values), "val", "tab:red"),
    ):
        if arr.size == 0:
            continue
        cdf = np.arange(1, arr.size + 1) / arr.size
        ax.plot(arr, cdf, label=label, color=color, lw=1.5)
    ax.set_xlabel(descriptor)
    ax.set_ylabel("CDF")
    p_str = f"{p_value:.4f}" if np.isfinite(p_value) else "nan"
    ax.set_title(f"{descriptor} — KS p = {p_str}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


__all__ = ["render_ks_cdf", "render_qc_grid"]
