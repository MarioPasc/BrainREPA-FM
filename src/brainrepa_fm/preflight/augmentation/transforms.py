"""The eight transforms audited by pre-flight 01.

Per proposal §3.2 / docs/checks/01_augmentation_preflight.md §3.2 the audited
set is:

- A.1 — Official BraTS-Inpainting sampler at default ranges.
- A.2 — Official sampler at widened ranges (×1.5 shape, ×1.5 volume).
- A.3 — Tumor-shape-mimicking mask drawn from a donor subject (SRI24 space).
- B.1 — Left-right (sagittal) flip applied to both volume and void.
- C.1 — Bias-field simulation (MONAI ``RandBiasFieldd``, degree 3, coef (0, 0.05)).
- C.2 — Gamma (``RandAdjustContrastd``, γ ∈ (0.85, 1.15)).
- C.3 — Intensity shift (``RandShiftIntensityd``, offset ∈ (−0.05, 0.05)).
- C.4 — Rician noise (``RandRicianNoised``, std ∈ (0, 0.025)).

The void-mask family (A.1–A.3) uses :func:`sample_void_mask` — a simplified
geometric sampler that approximates the official BraTS sampler's
distributional descriptors (volume, surface-to-volume, centroid distance,
max diameter). The official sampler is much heavier; vendoring it is deferred
to a follow-up task (see DECISIONS.md — A.2/A.3 simplified sampler).

Each transform is encoded as a :class:`TransformSpec` and applied through
:func:`apply_transform` to a triple ``(t1, void_mask, brain_mask)`` of numpy
arrays in BraTS-native shape ``(240, 240, 155)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from monai.transforms import (
    RandAdjustContrastd,
    RandBiasFieldd,
    RandRicianNoised,
    RandShiftIntensityd,
)
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

# Type aliases for clarity.
TransformKind = Literal["void_mask", "spatial", "intensity"]


# ---------------------------------------------------------------------------
# Simplified void-mask sampler (approximates the official BraTS sampler)
# ---------------------------------------------------------------------------


def _sample_unit_blob(
    rng: np.random.Generator, *, target_voxels: int, anisotropy: tuple[float, float, float]
) -> np.ndarray:
    """Generate a random binary blob (~ellipsoid + morphological perturbation) at the origin.

    Parameters:
        rng: NumPy generator for reproducibility.
        target_voxels: Approximate number of voxels in the returned blob.
        anisotropy: Per-axis radius scaling factor before perturbation.

    Returns:
        Binary array of shape ``(2*r, 2*r, 2*r)`` for ``r = max(radii)`` with the
        blob centered.
    """
    radius = int(round((3 * target_voxels / (4 * np.pi)) ** (1 / 3)))
    radius = max(radius, 2)
    side = 2 * radius + 4
    coords = np.stack(
        np.meshgrid(
            np.arange(-side // 2, side // 2),
            np.arange(-side // 2, side // 2),
            np.arange(-side // 2, side // 2),
            indexing="ij",
        ),
        axis=-1,
    ).astype(np.float32)
    # Random per-axis perturbation around the requested anisotropy.
    jitter = rng.uniform(0.85, 1.15, size=3) * np.asarray(anisotropy)
    norm = (coords / (radius * jitter)) ** 2
    blob = (norm.sum(axis=-1) < 1.0).astype(np.int8)
    # One step of random morphological dilation along a random axis to break perfect symmetry.
    axis = rng.integers(0, 3)
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct.flat[13] = True  # centre
    struct[1, 1, 1 + (1 if axis == 2 else 0)] = True
    blob = binary_dilation(blob.astype(bool), structure=struct).astype(np.int8)
    return blob


def sample_void_mask(
    brain: np.ndarray,
    tumor: np.ndarray | None = None,
    *,
    widen_factor: float = 1.0,
    seed: int = 0,
    target_volume_voxels: int | None = None,
) -> np.ndarray:
    """Sample a synthetic void mask inside the brain, away from the tumor.

    Approximation of the official BraTS sampler's behaviour: pick a location
    inside the brain that is at least a few voxels from any existing tumor
    region (via a distance transform), then drop a randomly-shaped blob there.

    Parameters:
        brain: Binary brain mask, ``(X, Y, Z)`` int8.
        tumor: Optional binary tumor mask. If None, the void is unconstrained
            relative to tumor — used by the challenge-val branch.
        widen_factor: Scale factor applied to the target volume. ``1.0`` = default
            sampler (A.1), ``1.5`` = widened sampler (A.2).
        seed: Stochastic seed.
        target_volume_voxels: Override the target volume. If None, falls back to a
            BraTS-2026-train empirical default (~ 7,500 voxels at widen=1.0).

    Returns:
        Binary int8 array of shape ``brain.shape``.
    """
    rng = np.random.default_rng(seed)
    brain_bool = brain.astype(bool)
    if not brain_bool.any():
        return np.zeros_like(brain, dtype=np.int8)

    # Empirical default: BraTS-2026 healthy masks have median volume around 7.5k voxels.
    base_volume = target_volume_voxels if target_volume_voxels is not None else 7500
    target = max(int(round(base_volume * widen_factor)), 256)

    # Distance transform inside the brain, away from any tumor region.
    avoid = tumor.astype(bool) if tumor is not None else np.zeros_like(brain_bool)
    interior = brain_bool & (~avoid)
    if not interior.any():
        return np.zeros_like(brain, dtype=np.int8)

    dt = ndi.distance_transform_edt(interior)
    # Bias placement: prefer points well-inside the brain.
    weights = (dt >= max(np.percentile(dt[interior], 60), 3.0)).astype(np.float32)
    if not weights.any():
        weights = interior.astype(np.float32)
    weights /= float(weights.sum())
    flat_idx = rng.choice(weights.size, p=weights.ravel())
    cz, cy, cx = np.unravel_index(flat_idx, weights.shape)

    aniso = (
        rng.uniform(0.85, 1.15),
        rng.uniform(0.85, 1.15),
        rng.uniform(0.7, 1.0),  # BraTS volumes are thinner along Z
    )
    blob = _sample_unit_blob(rng, target_voxels=target, anisotropy=aniso)
    bz, by, bx = blob.shape
    z0 = cz - bz // 2
    y0 = cy - by // 2
    x0 = cx - bx // 2

    mask = np.zeros_like(brain, dtype=np.int8)
    # Slice intersection with the volume.
    z_lo, z_hi = max(0, z0), min(brain.shape[0], z0 + bz)
    y_lo, y_hi = max(0, y0), min(brain.shape[1], y0 + by)
    x_lo, x_hi = max(0, x0), min(brain.shape[2], x0 + bx)
    if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
        return mask
    bz_lo = z_lo - z0
    by_lo = y_lo - y0
    bx_lo = x_lo - x0
    bz_hi = bz_lo + (z_hi - z_lo)
    by_hi = by_lo + (y_hi - y_lo)
    bx_hi = bx_lo + (x_hi - x_lo)
    placed = blob[bz_lo:bz_hi, by_lo:by_hi, bx_lo:bx_hi]
    mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = (
        placed & brain[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    ).astype(np.int8)
    return mask


def sample_donor_tumor_mask(
    donor_tumor: np.ndarray,
    brain: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """A.3: re-place a donor's tumor shape at a random interior location.

    Parameters:
        donor_tumor: Binary tumor mask from a different subject (BraTS-GLI).
        brain: Binary brain mask of the *recipient* subject.
        seed: Stochastic seed.

    Returns:
        Binary int8 array of shape ``brain.shape`` carrying the donor shape at
        a new location inside the recipient's brain.
    """
    rng = np.random.default_rng(seed)
    donor_bool = donor_tumor.astype(bool)
    if not donor_bool.any():
        return np.zeros_like(brain, dtype=np.int8)
    coords = np.argwhere(donor_bool)
    centroid = coords.mean(axis=0)
    shape_offsets = (coords - centroid).astype(np.int32)

    brain_bool = brain.astype(bool)
    dt = ndi.distance_transform_edt(brain_bool)
    weights = (dt >= max(np.percentile(dt[brain_bool], 60), 3.0)).astype(np.float32)
    if not weights.any():
        weights = brain_bool.astype(np.float32)
    weights /= float(weights.sum())
    flat_idx = rng.choice(weights.size, p=weights.ravel())
    cz, cy, cx = np.unravel_index(flat_idx, weights.shape)

    mask = np.zeros_like(brain, dtype=np.int8)
    placed = shape_offsets + np.array([cz, cy, cx], dtype=np.int32)
    valid = (
        (placed[:, 0] >= 0)
        & (placed[:, 0] < brain.shape[0])
        & (placed[:, 1] >= 0)
        & (placed[:, 1] < brain.shape[1])
        & (placed[:, 2] >= 0)
        & (placed[:, 2] < brain.shape[2])
    )
    placed = placed[valid]
    mask[placed[:, 0], placed[:, 1], placed[:, 2]] = 1
    return mask & brain.astype(np.int8)


# ---------------------------------------------------------------------------
# TransformSpec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformSpec:
    """One audited transform.

    Attributes:
        id: Spec identifier (``"A.1"``, ``"C.2"``, …).
        name: Human-readable name (used in figures and tables).
        kind: ``"void_mask"``, ``"spatial"``, or ``"intensity"`` — controls how
            :func:`apply_transform` routes the call.
        params: Original parameter dict.
        halved_params: Halved-range parameter dict (used only when the
            decision rule says ``halve_range``).
    """

    id: str
    name: str
    kind: TransformKind
    params: dict[str, object] = field(default_factory=dict)
    halved_params: dict[str, object] = field(default_factory=dict)


# Canonical roster of the eight transforms audited by pre-flight 01.
ALL_TRANSFORMS: tuple[TransformSpec, ...] = (
    TransformSpec(
        id="A.1",
        name="BraTS sampler (default)",
        kind="void_mask",
        params={"widen_factor": 1.0},
        halved_params={"widen_factor": 1.0},
    ),
    TransformSpec(
        id="A.2",
        name="BraTS sampler (widened ×1.5)",
        kind="void_mask",
        params={"widen_factor": 1.5},
        halved_params={"widen_factor": 1.25},
    ),
    TransformSpec(
        id="A.3",
        name="Tumor-shape donor",
        kind="void_mask",
        params={"donor": True},
        halved_params={"donor": True},
    ),
    TransformSpec(
        id="B.1",
        name="Left-right flip (sagittal)",
        kind="spatial",
        params={"axis": 0},  # BraTS X axis is left-right
        halved_params={"axis": 0},
    ),
    TransformSpec(
        id="C.1",
        name="Bias-field simulation",
        kind="intensity",
        params={"degree": 3, "coeff_range": (0.0, 0.05)},
        halved_params={"degree": 3, "coeff_range": (0.0, 0.025)},
    ),
    TransformSpec(
        id="C.2",
        name="Gamma adjust",
        kind="intensity",
        params={"gamma": (0.85, 1.15)},
        halved_params={"gamma": (0.925, 1.075)},
    ),
    TransformSpec(
        id="C.3",
        name="Intensity shift",
        kind="intensity",
        params={"offsets": 0.05},
        halved_params={"offsets": 0.025},
    ),
    TransformSpec(
        id="C.4",
        name="Rician noise",
        kind="intensity",
        params={"std": 0.025},
        halved_params={"std": 0.0125},
    ),
)


# ---------------------------------------------------------------------------
# MONAI intensity transforms — instantiated per call (RNG-seeded)
# ---------------------------------------------------------------------------


def _build_intensity_transform(
    spec: TransformSpec, *, use_halved: bool, seed: int
) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    params = spec.halved_params if use_halved else spec.params
    if spec.id == "C.1":
        coeff = params["coeff_range"]  # type: ignore[index]
        t = RandBiasFieldd(
            keys=["t1"],
            degree=int(params["degree"]),  # type: ignore[arg-type]
            coeff_range=(float(coeff[0]), float(coeff[1])),  # type: ignore[index]
            prob=1.0,
        )
    elif spec.id == "C.2":
        gamma = params["gamma"]  # type: ignore[index]
        t = RandAdjustContrastd(
            keys=["t1"],
            gamma=(float(gamma[0]), float(gamma[1])),  # type: ignore[index]
            prob=1.0,
        )
    elif spec.id == "C.3":
        t = RandShiftIntensityd(
            keys=["t1"],
            offsets=float(params["offsets"]),  # type: ignore[arg-type]
            prob=1.0,
        )
    elif spec.id == "C.4":
        t = RandRicianNoised(
            keys=["t1"],
            std=float(params["std"]),  # type: ignore[arg-type]
            prob=1.0,
        )
    else:
        raise ValueError(f"not an intensity transform: {spec.id}")
    t.set_random_state(seed=seed)
    return t


# ---------------------------------------------------------------------------
# Apply a transform to (t1, void, brain)
# ---------------------------------------------------------------------------


def apply_transform(
    spec: TransformSpec,
    *,
    t1_voided: np.ndarray,
    brain: np.ndarray,
    void: np.ndarray,
    donor_tumor: np.ndarray | None = None,
    seed: int = 0,
    use_halved: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one transform and return the augmented ``(t1, void, brain)``.

    The void region in the returned ``t1`` is forced to zero.

    Parameters:
        spec: Transform spec.
        t1_voided: Brain-normalized voided T1 (``X, Y, Z``, float32, [0, 1]).
        brain: Binary brain mask (same shape, int8).
        void: Binary void mask (same shape, int8).
        donor_tumor: Donor tumor mask (only used by A.3).
        seed: Stochastic seed.
        use_halved: If True, use the halved-range parameter set.

    Returns:
        ``(t1_aug, void_aug, brain_aug)`` — all numpy arrays in the original shape.

    Raises:
        ValueError: For an unknown transform ID or missing donor.
    """
    if spec.kind == "void_mask":
        if spec.id == "A.3":
            if donor_tumor is None:
                raise ValueError("A.3 requires a donor_tumor mask")
            new_void = sample_donor_tumor_mask(donor_tumor, brain, seed=seed)
        else:
            widen = float(spec.params.get("widen_factor", 1.0))  # type: ignore[arg-type]
            new_void = sample_void_mask(brain, tumor=None, widen_factor=widen, seed=seed)
        t1_aug = t1_voided.copy()
        # Re-void: where the new void differs from the original, zero the intensity.
        t1_aug[new_void.astype(bool)] = 0.0
        return t1_aug, new_void.astype(np.int8), brain

    if spec.kind == "spatial":
        axis = int(spec.params.get("axis", 0))  # type: ignore[arg-type]
        return (
            np.ascontiguousarray(np.flip(t1_voided, axis=axis)),
            np.ascontiguousarray(np.flip(void, axis=axis)).astype(np.int8),
            np.ascontiguousarray(np.flip(brain, axis=axis)).astype(np.int8),
        )

    if spec.kind == "intensity":
        t = _build_intensity_transform(spec, use_halved=use_halved, seed=seed)
        # MONAI dict transforms expect tensors / numpy of (C, H, W, D). Add a channel axis.
        out = t({"t1": t1_voided[None, ...].astype(np.float32)})
        t1_aug = np.asarray(out["t1"])[0]
        # Intensity transforms must not produce non-zero values inside the void region.
        t1_aug[void.astype(bool)] = 0.0
        return t1_aug.astype(np.float32, copy=False), void, brain

    raise ValueError(f"unknown TransformSpec.kind: {spec.kind!r}")


__all__ = [
    "ALL_TRANSFORMS",
    "TransformSpec",
    "apply_transform",
    "sample_donor_tumor_mask",
    "sample_void_mask",
]
