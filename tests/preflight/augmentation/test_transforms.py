"""Transforms + sampler unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from brainrepa_fm.preflight.augmentation.transforms import (
    ALL_TRANSFORMS,
    apply_transform,
    sample_donor_tumor_mask,
    sample_void_mask,
)


def _make_brain() -> np.ndarray:
    """Build a (240, 240, 155) ellipsoid brain mask."""
    xx, yy, zz = np.meshgrid(
        np.arange(240) - 120, np.arange(240) - 120, np.arange(155) - 78, indexing="ij"
    )
    return ((xx**2) / 100**2 + (yy**2) / 100**2 + (zz**2) / 60**2 < 1.0).astype(np.int8)


@pytest.mark.unit
def test_eight_transforms_registered() -> None:
    ids = [t.id for t in ALL_TRANSFORMS]
    assert ids == ["A.1", "A.2", "A.3", "B.1", "C.1", "C.2", "C.3", "C.4"]


@pytest.mark.unit
def test_sampler_volume_scales_with_widen_factor() -> None:
    brain = _make_brain()
    v1 = int(sample_void_mask(brain, tumor=None, widen_factor=1.0, seed=42).sum())
    v15 = int(sample_void_mask(brain, tumor=None, widen_factor=1.5, seed=42).sum())
    # ×1.5 widen should bump volume by roughly 30-80% (sampler is stochastic).
    assert v15 > v1
    assert v15 < 3.0 * v1


@pytest.mark.unit
def test_sampler_respects_brain_boundary() -> None:
    brain = _make_brain()
    mask = sample_void_mask(brain, tumor=None, widen_factor=1.0, seed=7)
    outside = mask & (~brain.astype(bool))
    assert outside.sum() == 0


@pytest.mark.unit
def test_donor_mask_inside_brain() -> None:
    brain = _make_brain()
    rng = np.random.default_rng(1)
    donor = np.zeros_like(brain)
    # A small ellipsoid donor tumor at the origin.
    coords = rng.integers(60, 100, size=(200, 3))
    donor[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    placed = sample_donor_tumor_mask(donor, brain, seed=3)
    assert placed.sum() > 0
    outside = placed & (~brain.astype(bool))
    assert outside.sum() == 0


@pytest.mark.unit
def test_intensity_transform_preserves_void() -> None:
    """Intensity transforms must keep the void region at zero."""
    brain = _make_brain()
    void = np.zeros_like(brain)
    void[60:80, 60:80, 60:80] = 1
    t1 = np.where(brain.astype(bool) & ~void.astype(bool), 0.5, 0.0).astype(np.float32)
    for spec in ALL_TRANSFORMS:
        if spec.kind != "intensity":
            continue
        t1_aug, void_aug, _ = apply_transform(spec, t1_voided=t1, brain=brain, void=void, seed=0)
        # Void region is still exactly zero.
        assert np.array_equal(void_aug, void)
        assert np.allclose(t1_aug[void.astype(bool)], 0.0)


@pytest.mark.unit
def test_spatial_flip_changes_orientation() -> None:
    """B.1 flips along axis 0; volumes are not identical to input."""
    brain = _make_brain()
    void = np.zeros_like(brain)
    void[60:80, 60:80, 60:80] = 1
    rng = np.random.default_rng(0)
    t1 = rng.random(brain.shape).astype(np.float32) * brain
    spec = next(t for t in ALL_TRANSFORMS if t.id == "B.1")
    t1_aug, void_aug, _ = apply_transform(spec, t1_voided=t1, brain=brain, void=void, seed=0)
    assert not np.array_equal(t1_aug, t1)
    # Flip is its own inverse.
    spec_again, _, _ = apply_transform(spec, t1_voided=t1_aug, brain=brain, void=void_aug, seed=0)
    np.testing.assert_array_equal(spec_again, t1)
