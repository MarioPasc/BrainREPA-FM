"""Unit tests for the pure reconstruction-metric computation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from brainrepa_fm.common.metrics import psnr
from brainrepa_fm.preflight.maisi_vae.reconstruction import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
)

pytestmark = [pytest.mark.unit, pytest.mark.preflight_maisi]


def _masks(shape: tuple[int, int, int] = (16, 16, 16)):
    brain = np.ones(shape, dtype=np.int8)
    tumor = np.zeros(shape, dtype=np.int8)
    tumor[4:8, 4:8, 4:8] = 1
    void = np.zeros(shape, dtype=np.int8)
    void[2:6, 2:6, 2:6] = 1
    return brain, tumor, void


def test_metrics_dataclass_is_frozen():
    m = ReconstructionMetrics("s", *([0.0] * 14))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.subject_id = "x"  # type: ignore[misc]


def test_identity_reconstruction_is_perfect():
    gt = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
    brain, tumor, void = _masks()
    res = compute_reconstruction_metrics(
        subject_id="s",
        gt=gt,
        recon=gt.copy(),
        brain_mask=brain,
        tumor_mask=tumor,
        void_masks=[void, void],
    )
    assert res.psnr_full == float("inf")
    assert res.psnr_void_mean == float("inf")
    assert res.ssim_full == pytest.approx(1.0, abs=1e-6)


def test_void_mean_is_the_mean_over_masks():
    rng = np.random.default_rng(1)
    gt = rng.random((16, 16, 16)).astype(np.float32)
    recon = np.clip(gt + rng.normal(0.0, 0.05, gt.shape), 0.0, 1.0).astype(np.float32)
    brain, tumor, _ = _masks()
    v1 = np.zeros((16, 16, 16), dtype=np.int8)
    v1[1:5, 1:5, 1:5] = 1
    v2 = np.zeros((16, 16, 16), dtype=np.int8)
    v2[8:12, 8:12, 8:12] = 1
    res = compute_reconstruction_metrics(
        subject_id="s",
        gt=gt,
        recon=recon,
        brain_mask=brain,
        tumor_mask=tumor,
        void_masks=[v1, v2],
    )
    expected = float(np.mean([psnr(gt, recon, v1), psnr(gt, recon, v2)]))
    assert res.psnr_void_mean == pytest.approx(expected, rel=1e-9)


def test_worst_void_psnr_is_not_better_than_mean():
    rng = np.random.default_rng(2)
    gt = rng.random((16, 16, 16)).astype(np.float32)
    recon = np.clip(gt + rng.normal(0.0, 0.08, gt.shape), 0.0, 1.0).astype(np.float32)
    brain, tumor, _ = _masks()
    voids = []
    for k in range(4):
        v = np.zeros((16, 16, 16), dtype=np.int8)
        v[k : k + 4, k : k + 4, k : k + 4] = 1
        voids.append(v)
    res = compute_reconstruction_metrics(
        subject_id="s",
        gt=gt,
        recon=recon,
        brain_mask=brain,
        tumor_mask=tumor,
        void_masks=voids,
    )
    assert res.psnr_void_worst <= res.psnr_void_mean
    assert res.mse_void_worst >= res.mse_void_mean


def test_shape_mismatch_raises():
    gt = np.zeros((16, 16, 16), dtype=np.float32)
    brain, tumor, void = _masks()
    with pytest.raises(ValueError):
        compute_reconstruction_metrics(
            subject_id="s",
            gt=gt,
            recon=np.zeros((8, 8, 8), dtype=np.float32),
            brain_mask=brain,
            tumor_mask=tumor,
            void_masks=[void],
        )
