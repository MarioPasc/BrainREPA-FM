"""Fit BraTS volumes to the MAISI VAE input envelope.

The MAISI-V2 VAE requires spatial dimensions divisible by its 4x downsampling
factor. BraTS-Inpainting volumes are ``(240, 240, 155)``; pre-flight 03 audits
at two envelopes (``docs/checks/03_maisi_vae_audit.md`` §1.3 / §3.1):

- 3060 / 4060 (memory-constrained): ``(192, 192, 144)`` — a center-crop.
- A100: ``(256, 256, 192)`` — a symmetric zero-pad that keeps the whole brain.

:func:`prepare_to_envelope` resolves both with one per-axis rule (crop axes
above the target, zero-pad axes below it), so it is correct for any
``(input, target)`` pair, not only the two BraTS cases.
"""

from __future__ import annotations

import numpy as np

__all__ = ["prepare_to_envelope"]


def prepare_to_envelope(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop axes above ``target_shape`` and zero-pad axes below it.

    The padded region carries no signal: ``pad_value=0`` is correct both for
    ``[0, 1]`` intensity volumes and for binary masks (brain / void / tumor).

    Parameters:
        volume: A 3-D ``(X, Y, Z)`` array — an intensity volume or a mask.
        target_shape: Desired ``(X, Y, Z)``.
        pad_value: Constant value for the padded region.

    Returns:
        An array of shape exactly ``target_shape``, dtype preserved.

    Raises:
        ValueError: If ``volume`` is not 3-D or ``target_shape`` is not length 3.
    """
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"prepare_to_envelope expects a 3-D array, got {arr.ndim}-D")
    if len(target_shape) != 3:
        raise ValueError(f"target_shape must be length 3, got {target_shape}")

    # Stage 1 — center-crop every axis that exceeds the target.
    crop: list[slice] = []
    for in_d, t_d in zip(arr.shape, target_shape, strict=True):
        if in_d > t_d:
            start = (in_d - t_d) // 2
            crop.append(slice(start, start + t_d))
        else:
            crop.append(slice(0, in_d))
    arr = arr[tuple(crop)]

    # Stage 2 — symmetric zero-pad every axis still below the target.
    pad: list[tuple[int, int]] = []
    for in_d, t_d in zip(arr.shape, target_shape, strict=True):
        diff = max(t_d - in_d, 0)
        pad.append((diff // 2, diff - diff // 2))
    if any(lo or hi for lo, hi in pad):
        arr = np.pad(arr, pad, mode="constant", constant_values=pad_value)
    return arr
