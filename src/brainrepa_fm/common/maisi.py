"""Frozen MAISI-V2 VAE-GAN wrapper for BrainREPA-FM.

This module is the canonical wrapper around the MONAI ``AutoencoderKlMaisi``
class. It loads the checkpoint at
``/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt``
with the network configuration distributed in
``MAISI_V2_RM/code/NV-Generate-CTMR/configs/config_network_ddpm.json`` and
exposes a deterministic ``encode`` (returns ``z_mu * scale_factor``) and a
matching ``decode`` (divides by ``scale_factor`` before forwarding).

Both pre-flight 01 (augmentation composability) and pre-flight 03 (VAE audit)
consume this module. The reparametrized sample ``encode_stage_2_inputs`` is
reserved for FM training and is **not** exposed here — see DECISIONS.md
(2026-05-19 — MAISI VAE audit uses z_mu).

Hard rules enforced by this module:

- The checkpoint and source tree are never written to.
- ``torch.no_grad()`` and ``.eval()`` apply to every forward pass.
- The checkpoint's SHA-256 is computed once at first load and logged at INFO.

The latent spatial shape produced by the upstream class for a given input
shape is not pinned at import time — it depends on the configured
``num_channels`` stack and the upstream ``AutoencoderKlMaisi``'s downsample
schedule. Pin it empirically with :func:`probe_latent_shape`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

logger = logging.getLogger(__name__)

DEFAULT_MAISI_CHECKPOINT: Path = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt"
)

DEFAULT_MAISI_CONFIG_PATH: Path = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR/configs/config_network_ddpm.json"
)

# Padding contracts. MAISI requires dimensions divisible by the downsample factor (4 = 2 ** 2,
# since num_channels=[64,128,256] gives 2 strided downsamples).
#
# - ``MAISI_PAD_SHAPE``: the full A100-class pad target. 256×256×192 → latent (4, 64, 64, 48).
#   Does NOT fit on the 12 GB RTX 3060 for a full encode→decode round-trip even with autocast.
#
# - ``MAISI_PAD_SHAPE_3060``: a memory-constrained pad target that fits on the 3060 in autocast
#   fp16 (peak ~7.3 GB round-trip). Latent shape (4, 48, 48, 36) for the 192×192×144 input. The
#   3060 path applies center-crop (240×240×155 → 192×192×144) to land here.
MAISI_PAD_SHAPE: tuple[int, int, int] = (256, 256, 192)
MAISI_PAD_SHAPE_3060: tuple[int, int, int] = (192, 192, 144)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _resolve_refs(value: object, top: dict[str, object]) -> object:
    """Resolve MONAI hydra-style ``@key`` references against the top-level dict."""
    if isinstance(value, str) and value.startswith("@"):
        key = value[1:]
        if key not in top:
            raise KeyError(f"unresolved hydra reference '{value}' (top-level keys: {list(top)})")
        return top[key]
    if isinstance(value, list):
        return [_resolve_refs(v, top) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, top) for k, v in value.items()}
    return value


def load_maisi_vae_config(config_path: Path | None = None) -> dict[str, object]:
    """Read the MAISI VAE network config JSON.

    Resolves the ``@image_channels`` / ``@latent_channels`` / ``@spatial_dims``
    references that MAISI ships with against the file's top-level keys, then
    strips MONAI hydra sentinels (``_target_``, ``_meta_``, …).

    Parameters:
        config_path: Path to ``config_network_ddpm.json``. Defaults to the
            canonical location declared in ``src/external/README.md``.

    Returns:
        The resolved ``autoencoder_def`` block as a kwargs dict ready for
        :class:`AutoencoderKlMaisi`.
    """
    if config_path is None:
        config_path = DEFAULT_MAISI_CONFIG_PATH
    with Path(config_path).open("r") as fh:
        raw = json.load(fh)
    if "autoencoder_def" in raw:
        block = raw["autoencoder_def"]
    elif "autoencoder" in raw:
        block = raw["autoencoder"]
    else:
        raise KeyError(
            f"{config_path} does not contain 'autoencoder_def' or 'autoencoder' block. "
            f"Top-level keys: {list(raw)[:8]}"
        )
    resolved = _resolve_refs(block, raw)
    if not isinstance(resolved, dict):
        raise TypeError("resolved autoencoder block must be a dict")
    return {k: v for k, v in resolved.items() if not k.startswith("_")}


def compute_vae_checkpoint_sha256(path: Path | None = None, *, prefix_chars: int = 16) -> str:
    """SHA-256 of the MAISI checkpoint file.

    Parameters:
        path: Path to ``autoencoder_v2.pt``.
        prefix_chars: How many hex chars to return (full hex if None).

    Returns:
        Hex digest, truncated to ``prefix_chars``.
    """
    if path is None:
        path = DEFAULT_MAISI_CHECKPOINT
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_chars]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaisiLoadInfo:
    """Bookkeeping returned by :func:`load_maisi_vae`."""

    checkpoint_path: Path
    config_path: Path
    sha256_prefix: str
    scale_factor: float
    config: dict[str, object]


class MaisiVAE(torch.nn.Module):
    """Deterministic wrapper around the frozen MAISI-V2 VAE-GAN.

    The wrapped network is loaded once at construction (no lazy load) and put
    into ``.eval()`` mode. All forward passes run under ``torch.no_grad()``.

    Attributes:
        autoencoder: The underlying :class:`AutoencoderKlMaisi`.
        scale_factor: Scalar from the checkpoint applied on encode
            (multiplied) and decode (divided).
        info: Bookkeeping (SHA-256 prefix, paths, config).
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        config_path: Path | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        autocast_fp16: bool = True,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        ckpt_path = (
            Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_MAISI_CHECKPOINT
        )
        cfg_path = Path(config_path) if config_path is not None else DEFAULT_MAISI_CONFIG_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MAISI checkpoint not found: {ckpt_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"MAISI config not found: {cfg_path}")

        config = load_maisi_vae_config(cfg_path)
        # MAISI ships ``norm_float16: True``, which forces group-norm to fp16 internally and
        # collides with fp32 input on consumer GPUs. Override to False; we obtain the same memory
        # benefit via torch.autocast at forward time.
        config["norm_float16"] = False
        # Activation checkpointing trades compute for VRAM — required for the round-trip on the 3060.
        config["use_checkpointing"] = use_checkpointing
        logger.info("instantiating AutoencoderKlMaisi with config keys=%s", sorted(config.keys()))
        self.autoencoder = AutoencoderKlMaisi(**config)  # type: ignore[arg-type]
        self._autocast_fp16 = autocast_fp16

        sha = compute_vae_checkpoint_sha256(ckpt_path)
        logger.info("loading MAISI checkpoint %s (sha256[:16]=%s)", ckpt_path, sha)
        # torch.load defaults changed across versions; map_location="cpu" is robust.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "unet_state_dict" in ckpt:
            state = ckpt["unet_state_dict"]
            scale_factor = float(ckpt.get("scale_factor", 1.0))
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            scale_factor = float(ckpt.get("scale_factor", 1.0))
        else:
            state = ckpt
            scale_factor = 1.0
        missing, unexpected = self.autoencoder.load_state_dict(state, strict=False)
        if missing:
            logger.warning("MAISI load: %d missing keys (first: %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning(
                "MAISI load: %d unexpected keys (first: %s)", len(unexpected), unexpected[:3]
            )

        self.scale_factor = scale_factor
        self.info = MaisiLoadInfo(
            checkpoint_path=ckpt_path,
            config_path=cfg_path,
            sha256_prefix=sha,
            scale_factor=scale_factor,
            config=config,
        )
        self.autoencoder.eval()
        self.autoencoder.to(device=device, dtype=dtype)
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)

    def _autocast(self):
        """Context manager: fp16 autocast iff ``self._autocast_fp16`` is True."""
        if self._autocast_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        # nullcontext-equivalent for non-CUDA / non-autocast runs.
        import contextlib

        return contextlib.nullcontext()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encode: returns ``z_mu * scale_factor``.

        Parameters:
            x: Input volume tensor of shape ``(B, 1, Z, Y, X)``.

        Returns:
            Latent tensor of shape ``(B, C, Lz, Ly, Lx)``. Dtype follows the
            autocast region — fp16 when ``autocast_fp16=True``.
        """
        with self._autocast():
            z_mu, _z_sigma = self.autoencoder.encode(x)
            return z_mu * self.scale_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent back to volume space.

        Parameters:
            z: Latent of shape ``(B, C, Lz, Ly, Lx)``, scaled by ``scale_factor``
                (as produced by :meth:`encode`).

        Returns:
            Decoded volume of shape ``(B, 1, Z, Y, X)``.
        """
        with self._autocast():
            return self.autoencoder.decode(z / self.scale_factor)

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: deterministic encode → decode round-trip in a single autocast region."""
        with self._autocast():
            z_mu, _ = self.autoencoder.encode(x)
            z = z_mu * self.scale_factor
            return self.autoencoder.decode(z / self.scale_factor)


# ---------------------------------------------------------------------------
# Padding / unpadding helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PadOffsets:
    """Per-axis (before, after) zero-padding amounts."""

    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]


def pad_to_maisi(
    x: torch.Tensor, target_shape: tuple[int, int, int] = MAISI_PAD_SHAPE
) -> tuple[torch.Tensor, PadOffsets]:
    """Symmetric-ish zero-pad a ``(..., Z, Y, X)`` tensor up to ``target_shape``.

    Parameters:
        x: Tensor with at least 3 spatial axes; the last three are padded.
        target_shape: Target ``(Z, Y, X)``. Must be ≥ x.shape[-3:].

    Returns:
        Padded tensor and the offsets needed to unpad.
    """
    z_in, y_in, x_in = x.shape[-3:]
    z_t, y_t, x_t = target_shape
    if z_in > z_t or y_in > y_t or x_in > x_t:
        raise ValueError(f"target_shape {target_shape} smaller than input {(z_in, y_in, x_in)}")

    def split(diff: int) -> tuple[int, int]:
        return (diff // 2, diff - diff // 2)

    pz = split(z_t - z_in)
    py = split(y_t - y_in)
    px = split(x_t - x_in)

    # torch.nn.functional.pad takes (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    pad = (px[0], px[1], py[0], py[1], pz[0], pz[1])
    out = torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
    return out, PadOffsets(z=pz, y=py, x=px)


def unpad_from_maisi(x_padded: torch.Tensor, offsets: PadOffsets) -> torch.Tensor:
    """Inverse of :func:`pad_to_maisi`."""
    z_lo, z_hi = offsets.z
    y_lo, y_hi = offsets.y
    x_lo, x_hi = offsets.x
    z_dim, y_dim, x_dim = x_padded.shape[-3:]
    return x_padded[
        ...,
        z_lo : z_dim - z_hi,
        y_lo : y_dim - y_hi,
        x_lo : x_dim - x_hi,
    ]


@dataclass(frozen=True)
class CropOffsets:
    """Per-axis ``(start, stop)`` slice bounds used by :func:`center_crop_to_maisi`."""

    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]


def center_crop_to_maisi(
    x: torch.Tensor, target_shape: tuple[int, int, int] = MAISI_PAD_SHAPE_3060
) -> tuple[torch.Tensor, CropOffsets]:
    """Symmetric center-crop a ``(..., Z, Y, X)`` tensor down to ``target_shape``.

    Used to land BraTS 240×240×155 inputs at the 192×192×144 envelope that fits
    on the 12 GB RTX 3060. Crop bounds are deterministic.

    Parameters:
        x: Input tensor.
        target_shape: Desired ``(Z, Y, X)``; each axis must be ≤ x.shape[-3:].

    Returns:
        Cropped tensor and the per-axis (start, stop) slice bounds.
    """
    z_in, y_in, x_in = x.shape[-3:]
    z_t, y_t, x_t = target_shape
    if z_in < z_t or y_in < y_t or x_in < x_t:
        raise ValueError(
            f"target_shape {target_shape} larger than input {(z_in, y_in, x_in)} on some axis"
        )

    def bounds(in_dim: int, out_dim: int) -> tuple[int, int]:
        start = (in_dim - out_dim) // 2
        return (start, start + out_dim)

    bz = bounds(z_in, z_t)
    by = bounds(y_in, y_t)
    bx = bounds(x_in, x_t)
    out = x[..., bz[0] : bz[1], by[0] : by[1], bx[0] : bx[1]]
    return out, CropOffsets(z=bz, y=by, x=bx)


# ---------------------------------------------------------------------------
# Empirical latent-shape probe
# ---------------------------------------------------------------------------


def probe_latent_shape(
    vae: MaisiVAE,
    *,
    input_shape: tuple[int, int, int] = MAISI_PAD_SHAPE_3060,
    device: torch.device | str = "cuda",
) -> tuple[int, int, int, int]:
    """Run one deterministic forward pass to recover the latent shape.

    Parameters:
        vae: A live :class:`MaisiVAE` instance.
        input_shape: ``(Z, Y, X)`` of the probe input. Defaults to the
            memory-constrained 3060 pad target. Pass :data:`MAISI_PAD_SHAPE`
            for an A100 run.
        device: Device on which to run the probe.

    Returns:
        ``(C, Lz, Ly, Lx)`` — the latent shape of one encoded sample.
    """
    probe = torch.zeros((1, 1, *input_shape), device=device, dtype=torch.float32)
    z = vae.encode(probe)
    if z.ndim != 5:
        raise RuntimeError(f"unexpected latent ndim {z.ndim}; full shape={tuple(z.shape)}")
    _, c, lz, ly, lx = z.shape
    return (int(c), int(lz), int(ly), int(lx))


# ---------------------------------------------------------------------------
# Volume → tensor convenience for offline use
# ---------------------------------------------------------------------------


def volume_to_tensor(x: np.ndarray, *, device: torch.device | str = "cuda") -> torch.Tensor:
    """Cast a ``(X, Y, Z)`` numpy array to a ``(1, 1, X, Y, Z)`` torch tensor on device.

    No axis permutation. MAISI is spatially symmetric — the three spatial axes
    are treated uniformly — so we keep BraTS's native (X, Y, Z) order.
    """
    if x.ndim != 3:
        raise ValueError(f"expected 3-D ndarray, got shape {x.shape}")
    arr = x[None, None, ...]
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(device)


def tensor_to_volume(t: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`volume_to_tensor`. Accepts ``(1, 1, X, Y, Z)`` or ``(X, Y, Z)``."""
    arr = t.detach().to(dtype=torch.float32, device="cpu").numpy()
    if arr.ndim == 5:
        return arr[0, 0]
    if arr.ndim == 4:
        return arr[0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"unexpected ndim {arr.ndim}")


__all__ = [
    "DEFAULT_MAISI_CHECKPOINT",
    "DEFAULT_MAISI_CONFIG_PATH",
    "MAISI_PAD_SHAPE",
    "MAISI_PAD_SHAPE_3060",
    "CropOffsets",
    "MaisiLoadInfo",
    "MaisiVAE",
    "PadOffsets",
    "center_crop_to_maisi",
    "compute_vae_checkpoint_sha256",
    "load_maisi_vae_config",
    "pad_to_maisi",
    "probe_latent_shape",
    "tensor_to_volume",
    "unpad_from_maisi",
    "volume_to_tensor",
]
