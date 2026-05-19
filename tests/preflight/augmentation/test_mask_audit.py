"""Mask-descriptor + KS test unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from brainrepa_fm.preflight.augmentation.mask_audit import (
    HARD_FAIL_DESCRIPTORS,
    MASK_DESCRIPTORS,
    compute_mask_descriptors,
    decide_hard_fail,
    ks_test,
)


def _make_brain() -> np.ndarray:
    xx, yy, zz = np.meshgrid(
        np.arange(60) - 30, np.arange(60) - 30, np.arange(60) - 30, indexing="ij"
    )
    return ((xx**2 + yy**2 + zz**2) < 25**2).astype(np.int8)


@pytest.mark.unit
def test_empty_mask_descriptors_are_zero() -> None:
    brain = _make_brain()
    mask = np.zeros_like(brain)
    desc = compute_mask_descriptors(mask, brain)
    assert desc == {n: 0.0 for n in MASK_DESCRIPTORS}


@pytest.mark.unit
def test_descriptor_volume_matches_count() -> None:
    brain = _make_brain()
    mask = np.zeros_like(brain)
    mask[10:14, 10:14, 10:14] = 1
    desc = compute_mask_descriptors(mask, brain)
    assert desc["volume"] == 64.0
    assert desc["surface_to_volume"] > 0.0
    assert desc["centroid_distance"] > 0.0
    assert desc["max_diameter"] > 0.0


@pytest.mark.unit
def test_ks_identical_distributions_high_p() -> None:
    rng = np.random.default_rng(0)
    train = {n: rng.normal(size=500) for n in MASK_DESCRIPTORS}
    val = {n: rng.normal(size=500) for n in MASK_DESCRIPTORS}
    ps = ks_test(train, val)
    for n in MASK_DESCRIPTORS:
        assert ps[n] > 0.05
    assert not decide_hard_fail(ps)


@pytest.mark.unit
def test_ks_shifted_distributions_hard_fail() -> None:
    rng = np.random.default_rng(0)
    train = {n: rng.normal(loc=0.0, size=500) for n in MASK_DESCRIPTORS}
    val = {n: rng.normal(loc=2.0, size=500) for n in MASK_DESCRIPTORS}
    ps = ks_test(train, val)
    for n in MASK_DESCRIPTORS:
        assert ps[n] < 0.05
    assert decide_hard_fail(ps)


@pytest.mark.unit
def test_decide_hard_fail_only_volume_or_centroid_triggers() -> None:
    """Surface-to-volume or max_diameter alone do NOT trigger hard fail."""
    ps_all_safe = {n: 0.2 for n in MASK_DESCRIPTORS}
    assert not decide_hard_fail(ps_all_safe)

    ps_only_sv_fail = dict(ps_all_safe, surface_to_volume=0.001)
    assert not decide_hard_fail(ps_only_sv_fail)

    ps_volume_fail = dict(ps_all_safe, volume=0.001)
    assert decide_hard_fail(ps_volume_fail)
    assert "volume" in HARD_FAIL_DESCRIPTORS
