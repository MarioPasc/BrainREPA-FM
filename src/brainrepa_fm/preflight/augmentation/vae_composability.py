"""Compute ``Δ_aug-VAE`` for one (transform, scan) pair.

Definition (proposal §3.4 / docs/checks/01_augmentation_preflight.md §3.4):

    Δ_aug-VAE(T_k; x) = PSNR(D(E(x)), x) − PSNR(D(E(T_k(x))), T_k(x))

PSNR is computed in three regions per pair:

- ``full``: the full padded volume (or the cropped 3060 envelope).
- ``brain``: inside ``masks/brain``.
- ``void``: inside ``masks/void`` (the inpainting region — the canonical signal).

For each transform we aggregate the per-scan Δ across N volumes into mean,
median, p10, and p90, per region. The median inside ``m_v`` drives the
include / halve / drop decision rule (§3.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from brainrepa_fm.common.maisi import (
    MaisiVAE,
    tensor_to_volume,
    volume_to_tensor,
)
from brainrepa_fm.preflight.augmentation.transforms import (
    TransformSpec,
    apply_transform,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaResult:
    """One transform / one scan result.

    Attributes:
        transform_id: e.g. ``"C.2"``.
        scan_id: e.g. ``"BraTS-GLI-00007-000"``.
        baseline_psnr: ``PSNR(D(E(x)), x)`` per region (``full`` / ``brain`` / ``void``).
        aug_psnr: ``PSNR(D(E(T(x))), T(x))`` per region.
        delta: ``baseline_psnr - aug_psnr`` per region (dB; positive = augmentation hurts VAE).
        latent_ratio: ``‖E(T(x)) − E(x)‖₂ / ‖E(x)‖₂`` — used by the C.4 "VAE erased noise" rule.
    """

    transform_id: str
    scan_id: str
    baseline_psnr: dict[str, float]
    aug_psnr: dict[str, float]
    delta: dict[str, float]
    latent_ratio: float


def _psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR in dB on a ``[0, 1]``-valued signal, optionally restricted to a mask."""
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
        diff = (prediction[m] - reference[m]).astype(np.float64)
    else:
        diff = (prediction - reference).astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0:
        return float("inf")
    return 10.0 * float(np.log10(1.0 / mse))


def _crop_to_envelope(
    t1: np.ndarray, brain: np.ndarray, void: np.ndarray, target_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric center-crop the three volumes to ``target_shape`` (matched to the VAE envelope)."""
    in_shape = t1.shape
    if in_shape == target_shape:
        return t1, brain, void

    def bounds(in_dim: int, out_dim: int) -> tuple[int, int]:
        if in_dim < out_dim:
            raise ValueError(f"cannot center-crop {in_dim} → {out_dim}")
        start = (in_dim - out_dim) // 2
        return (start, start + out_dim)

    bx = bounds(in_shape[0], target_shape[0])
    by = bounds(in_shape[1], target_shape[1])
    bz = bounds(in_shape[2], target_shape[2])
    sl = (slice(bx[0], bx[1]), slice(by[0], by[1]), slice(bz[0], bz[1]))
    return t1[sl], brain[sl], void[sl]


def _encode_decode_np(
    vae: MaisiVAE, x: np.ndarray, *, device: torch.device | str
) -> tuple[np.ndarray, np.ndarray]:
    """Run encode → decode on a single ``(X, Y, Z)`` numpy volume.

    Returns:
        (decoded volume, encoded latent) — both numpy arrays.
    """
    t = volume_to_tensor(x, device=device)
    z = vae.encode(t)
    y = vae.decode(z)
    y_np = tensor_to_volume(y.float())
    z_np = z.detach().to(dtype=torch.float32, device="cpu").numpy()
    del t, y, z
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return y_np, z_np


def compute_delta_aug_vae(
    vae: MaisiVAE,
    *,
    scan_id: str,
    t1_voided: np.ndarray,
    brain: np.ndarray,
    void: np.ndarray,
    transform: TransformSpec,
    donor_tumor: np.ndarray | None = None,
    seed: int = 0,
    use_halved: bool = False,
    target_shape: tuple[int, int, int] = (192, 192, 144),
    device: torch.device | str = "cuda",
) -> DeltaResult:
    """Compute one :class:`DeltaResult` for one (transform, scan).

    Parameters:
        vae: Live :class:`MaisiVAE` instance.
        scan_id: Identifier persisted in the result.
        t1_voided: Brain-normalized voided T1 ``(X, Y, Z)`` float32.
        brain: Binary brain mask same shape (int8).
        void: Binary void mask same shape (int8).
        transform: One :class:`TransformSpec`.
        donor_tumor: Required iff ``transform.id == "A.3"``.
        seed: Stochastic seed forwarded to the transform.
        use_halved: If True, use the transform's halved-range parameters.
        target_shape: VAE input envelope (``(192, 192, 144)`` on the 3060,
            ``(256, 256, 192)`` on the A100).
        device: CUDA device for the encode/decode passes.

    Returns:
        A :class:`DeltaResult`.
    """
    # 1. Center-crop everything to the VAE envelope BEFORE augmentation; mask sampling
    #    happens inside the cropped brain so the void lives in the same volume.
    t1_c, brain_c, void_c = _crop_to_envelope(t1_voided, brain, void, target_shape)
    donor_c: np.ndarray | None = None
    if donor_tumor is not None:
        donor_c, _, _ = _crop_to_envelope(donor_tumor, brain, void, target_shape)

    # 2. Baseline round-trip (no augmentation).
    decoded_x, z_x = _encode_decode_np(vae, t1_c, device=device)

    # 3. Apply the transform and run a second round-trip.
    t1_aug, void_aug, brain_aug = apply_transform(
        transform,
        t1_voided=t1_c,
        brain=brain_c,
        void=void_c,
        donor_tumor=donor_c,
        seed=seed,
        use_halved=use_halved,
    )
    decoded_aug, z_aug = _encode_decode_np(vae, t1_aug, device=device)

    # 4. PSNR per region.
    regions: dict[str, np.ndarray | None] = {
        "full": None,
        "brain": brain_c.astype(bool),
        "void": void_c.astype(bool),
    }
    baseline_psnr = {region: _psnr(t1_c, decoded_x, mask=m) for region, m in regions.items()}
    regions_aug: dict[str, np.ndarray | None] = {
        "full": None,
        "brain": brain_aug.astype(bool),
        "void": void_aug.astype(bool),
    }
    aug_psnr = {region: _psnr(t1_aug, decoded_aug, mask=m) for region, m in regions_aug.items()}
    delta = {region: float(baseline_psnr[region] - aug_psnr[region]) for region in regions}

    # 5. Latent ratio for the C.4 special drop rule.
    denom = float(np.linalg.norm(z_x.astype(np.float64).ravel()) + 1e-12)
    numer = float(np.linalg.norm((z_aug - z_x).astype(np.float64).ravel()))
    latent_ratio = numer / denom

    return DeltaResult(
        transform_id=transform.id,
        scan_id=scan_id,
        baseline_psnr=baseline_psnr,
        aug_psnr=aug_psnr,
        delta=delta,
        latent_ratio=latent_ratio,
    )


def aggregate_deltas(
    results: list[DeltaResult],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-scan ΔPSNR across volumes.

    Parameters:
        results: List of :class:`DeltaResult`. May span multiple transforms.

    Returns:
        Nested dict ``{transform_id: {region: {"mean", "median", "p10", "p90"}}}``.
    """
    by_transform: dict[str, list[DeltaResult]] = {}
    for r in results:
        by_transform.setdefault(r.transform_id, []).append(r)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for tid, group in by_transform.items():
        region_stats: dict[str, dict[str, float]] = {}
        for region in ("full", "brain", "void"):
            vals = np.array([r.delta[region] for r in group], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                region_stats[region] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "p10": float("nan"),
                    "p90": float("nan"),
                }
                continue
            region_stats[region] = {
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "p10": float(np.percentile(finite, 10)),
                "p90": float(np.percentile(finite, 90)),
            }
        out[tid] = region_stats
    return out


__all__ = [
    "DeltaResult",
    "aggregate_deltas",
    "compute_delta_aug_vae",
]
