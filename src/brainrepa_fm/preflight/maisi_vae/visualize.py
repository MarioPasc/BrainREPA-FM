"""Figures for pre-flight 03 — the MAISI VAE reconstruction audit.

All figures are written as PNG at 200 dpi with the headless ``Agg`` backend.
The equivariance figures listed in ``docs/checks/03`` §8 are intentionally
absent — the equivariance audit is deferred (see ``DECISIONS.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "render_latent_stats_figure",
    "render_psnr_histogram",
    "render_reconstruction_montage",
    "render_ssim_histogram",
    "render_voided_scatter",
]


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def _slice_axial(vol: np.ndarray) -> np.ndarray:
    return vol[:, :, vol.shape[2] // 2]


def _slice_sagittal(vol: np.ndarray) -> np.ndarray:
    return vol[vol.shape[0] // 2, :, :]


def _slice_coronal(vol: np.ndarray) -> np.ndarray:
    return vol[:, vol.shape[1] // 2, :]


def render_psnr_histogram(
    values: Sequence[float],
    *,
    region_label: str,
    threshold_db: float | None = None,
    out_path: Path,
) -> Path:
    """Per-volume PSNR histogram for one region.

    Parameters:
        values: Per-volume PSNR values (dB); non-finite entries are dropped.
        region_label: Region name for the title (e.g. ``"inside void"``).
        threshold_db: Optional decision threshold drawn as a dashed line.
        out_path: PNG destination (parent must exist).

    Returns:
        ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    finite = _finite(values)
    if finite.size:
        ax.hist(finite, bins=24, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(
            float(np.median(finite)),
            color="navy",
            lw=1.5,
            label=f"median {np.median(finite):.2f} dB",
        )
    if threshold_db is not None:
        ax.axvline(
            float(threshold_db),
            color="firebrick",
            ls="--",
            lw=1.5,
            label=f"threshold {threshold_db:.0f} dB",
        )
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("count")
    ax.set_title(f"Round-trip PSNR — {region_label}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def render_ssim_histogram(
    values: Sequence[float],
    *,
    region_label: str,
    out_path: Path,
) -> Path:
    """Per-volume SSIM histogram for one region.

    Parameters:
        values: Per-volume SSIM values; non-finite entries are dropped.
        region_label: Region name for the title.
        out_path: PNG destination.

    Returns:
        ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    finite = _finite(values)
    if finite.size:
        ax.hist(finite, bins=24, color="seagreen", edgecolor="black", alpha=0.85)
        ax.axvline(
            float(np.median(finite)),
            color="darkgreen",
            lw=1.5,
            label=f"median {np.median(finite):.4f}",
        )
        ax.legend(loc="best")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("count")
    ax.set_title(f"Round-trip SSIM — {region_label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def render_reconstruction_montage(
    *,
    gt_volume: np.ndarray,
    reconstructed: np.ndarray,
    void_mask: np.ndarray,
    label: str,
    out_path: Path,
) -> Path:
    """3x3 montage: axial / sagittal / coronal x original / reconstruction / residual.

    Parameters:
        gt_volume: Intact volume ``(X, Y, Z)`` in ``[0, 1]``.
        reconstructed: Round-trip ``D(E(x))``, same shape.
        void_mask: A representative void mask, outlined on the residual row.
        label: ``"best"`` / ``"median"`` / ``"worst"`` (or any caption).
        out_path: PNG destination.

    Returns:
        ``out_path``.
    """
    gt_volume = np.asarray(gt_volume)
    reconstructed = np.asarray(reconstructed)
    void_mask = np.asarray(void_mask)
    slicers = (_slice_axial, _slice_sagittal, _slice_coronal)
    titles = ("axial", "sagittal", "coronal")
    diff = np.abs(reconstructed - gt_volume)
    vmax = float(np.percentile(diff, 99)) or 1.0

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(f"reconstruction — {label}", fontsize=13)
    for col, (sl, title) in enumerate(zip(slicers, titles, strict=True)):
        axes[0, col].imshow(sl(gt_volume).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"x ({title})")
        axes[1, col].imshow(sl(reconstructed).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"D(E(x)) ({title})")
        ax2 = axes[2, col]
        ax2.imshow(sl(diff).T, cmap="hot", origin="lower", vmin=0, vmax=vmax)
        void_slice = sl(void_mask)
        if void_slice.any():
            ax2.contour(void_slice.T, levels=[0.5], colors="cyan", linewidths=0.8)
        ax2.set_title(f"|residual| ({title})")
        ax2.set_facecolor("black")
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def render_latent_stats_figure(
    mean: Sequence[float],
    std: Sequence[float],
    *,
    out_path: Path,
) -> Path:
    """Per-channel latent mean / std bar chart with ``N(0, 1)`` reference bands.

    Parameters:
        mean: Per-channel latent mean.
        std: Per-channel latent std.
        out_path: PNG destination.

    Returns:
        ``out_path``.
    """
    mean_a = np.asarray(list(mean), dtype=np.float64)
    std_a = np.asarray(list(std), dtype=np.float64)
    channels = np.arange(mean_a.size)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.bar(channels, mean_a, color="slateblue", edgecolor="black")
    ax0.axhline(0.0, color="black", lw=1)
    for ref in (-0.2, 0.2):
        ax0.axhline(ref, color="firebrick", ls="--", lw=1)
    ax0.set_title("per-channel latent mean (target |μ| < 0.2)")
    ax0.set_xlabel("latent channel")
    ax0.set_ylabel("mean")
    ax0.set_xticks(channels)

    ax1.bar(channels, std_a, color="slateblue", edgecolor="black")
    ax1.axhline(1.0, color="black", lw=1)
    for ref in (0.5, 1.5):
        ax1.axhline(ref, color="firebrick", ls="--", lw=1)
    ax1.set_title("per-channel latent std (target |sigma-1| < 0.5)")
    ax1.set_xlabel("latent channel")
    ax1.set_ylabel("std")
    ax1.set_xticks(channels)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def render_voided_scatter(
    s_inside: Sequence[float],
    s_outside: Sequence[float],
    *,
    out_path: Path,
) -> Path:
    """Scatter of §7 ``S_outside`` (x) vs ``S_inside`` (y), one point per volume.

    Parameters:
        s_inside: Per-volume mean ``S_inside``.
        s_outside: Per-volume mean ``S_outside``.
        out_path: PNG destination.

    Returns:
        ``out_path``.
    """
    si = np.asarray(list(s_inside), dtype=np.float64)
    so = np.asarray(list(s_outside), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(so, si, c="darkorange", edgecolor="black", alpha=0.8)
    ax.set_xlabel("S_outside  (want ≈ 0 — encoder locality)")
    ax.set_ylabel("S_inside  (want large — void/content separable)")
    ax.set_title("§7 voided vs non-voided encoder behaviour")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)
