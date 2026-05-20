"""Unit tests for the §7 voided-encoder behaviour computation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from brainrepa_fm.preflight.maisi_vae.voided_tests import (
    VoidedTestResult,
    compute_voided_tests_from_latents,
    downsample_mask_to_latent,
)

pytestmark = [pytest.mark.unit, pytest.mark.preflight_maisi]


def test_downsample_matches_latent_shape():
    mask = np.zeros((192, 192, 144), dtype=np.int8)
    mask[0:40, 0:40, 0:40] = 1
    out = downsample_mask_to_latent(mask, (48, 48, 36))
    assert out.shape == (48, 48, 36)
    assert out.dtype == bool
    assert out.any()


def test_downsample_marks_block_if_any_voxel_inside():
    mask = np.zeros((16, 16, 16), dtype=np.int8)
    mask[0, 0, 0] = 1
    out = downsample_mask_to_latent(mask, (4, 4, 4))
    assert out[0, 0, 0]
    assert int(out.sum()) == 1


def test_identical_latents_give_zero_energy():
    z = np.random.default_rng(0).normal(size=(1, 4, 8, 8, 8)).astype(np.float32)
    m = np.zeros((8, 8, 8), dtype=bool)
    m[2:5, 2:5, 2:5] = True
    res = compute_voided_tests_from_latents(
        subject_id="s", z_gt=z, z_voided=z.copy(), latent_void_masks=[m]
    )
    assert res.s_inside_mean == pytest.approx(0.0)
    assert res.s_outside_mean == pytest.approx(0.0)


def test_inside_only_perturbation_localises():
    rng = np.random.default_rng(1)
    z_gt = rng.normal(size=(1, 4, 8, 8, 8)).astype(np.float32)
    m = np.zeros((8, 8, 8), dtype=bool)
    m[2:5, 2:5, 2:5] = True
    z_void = z_gt.copy()
    z_void[:, :, m] += 5.0  # perturb only inside the latent void
    res = compute_voided_tests_from_latents(
        subject_id="s", z_gt=z_gt, z_voided=z_void, latent_void_masks=[m]
    )
    assert res.s_inside_mean > 1.0
    assert res.s_outside_mean == pytest.approx(0.0)


def test_result_dataclass_is_frozen():
    r = VoidedTestResult("s", 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.subject_id = "x"  # type: ignore[misc]
