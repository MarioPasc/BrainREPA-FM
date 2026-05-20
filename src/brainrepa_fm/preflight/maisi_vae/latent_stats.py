"""Per-channel latent statistics for the MAISI VAE audit (Caveat 8).

``docs/checks/03_maisi_vae_audit.md`` Caveat 8: the MAISI-v2 paper assumes
latent values follow ``N(0, 1)`` per channel. This module accumulates the
per-channel mean and standard deviation of the (already
``scale_factor``-multiplied) latents across the audited cohort using Chan's
parallel-variance combination, so the full latent stack never has to be held
in memory.

The result populates the Schema B placeholders ``latent_mean`` and
``latent_scale`` (both ``(4,)``) and is reported in ``decision.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["LatentChannelStats", "LatentStatsAccumulator"]


@dataclass(frozen=True)
class LatentChannelStats:
    """Per-channel latent mean / std pooled over the cohort.

    Both tuples have one entry per latent channel; statistics pool every
    spatial position of every encoded volume.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]


class LatentStatsAccumulator:
    """Streaming per-channel mean/variance over a sequence of latent tensors.

    Each :meth:`update` folds one ``(B, C, Lz, Ly, Lx)`` latent into the running
    statistics; :meth:`result` returns the pooled population mean and std.
    """

    def __init__(self, n_channels: int) -> None:
        """Parameters: ``n_channels`` — latent channel count (4 for MAISI-v2)."""
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        self._count = np.zeros(n_channels, dtype=np.float64)
        self._mean = np.zeros(n_channels, dtype=np.float64)
        self._m2 = np.zeros(n_channels, dtype=np.float64)

    def update(self, z: torch.Tensor | np.ndarray) -> None:
        """Fold one latent tensor ``(B, C, Lz, Ly, Lx)`` into the running stats.

        Raises:
            ValueError: If ``z`` is not 5-D or its channel count disagrees with
                the value passed to :meth:`__init__`.
        """
        if isinstance(z, torch.Tensor):
            arr = z.detach().to(dtype=torch.float32, device="cpu").numpy()
        else:
            arr = np.asarray(z, dtype=np.float32)
        if arr.ndim != 5:
            raise ValueError(f"expected a 5-D (B,C,Lz,Ly,Lx) latent, got {arr.ndim}-D")
        if arr.shape[1] != self._count.shape[0]:
            raise ValueError(
                f"channel mismatch: latent has {arr.shape[1]}, accumulator has "
                f"{self._count.shape[0]}"
            )
        # (B, C, ...) -> (C, N): each channel is an independent sample stream.
        per_channel = np.moveaxis(arr, 1, 0).reshape(arr.shape[1], -1).astype(np.float64)
        n_b = per_channel.shape[1]
        if n_b == 0:
            return
        mean_b = per_channel.mean(axis=1)
        m2_b = ((per_channel - mean_b[:, None]) ** 2).sum(axis=1)

        # Chan et al. parallel combination of (count, mean, M2).
        delta = mean_b - self._mean
        new_count = self._count + n_b
        safe = np.where(new_count > 0, new_count, 1.0)
        self._mean = self._mean + delta * (n_b / safe)
        self._m2 = self._m2 + m2_b + (delta**2) * (self._count * n_b / safe)
        self._count = new_count

    def result(self) -> LatentChannelStats:
        """Return the pooled per-channel mean and population std."""
        with np.errstate(invalid="ignore", divide="ignore"):
            safe = np.where(self._count > 0, self._count, 1.0)
            var = np.where(self._count > 0, self._m2 / safe, np.nan)
        std = np.sqrt(var)
        return LatentChannelStats(
            mean=tuple(float(x) for x in self._mean),
            std=tuple(float(x) for x in std),
        )
