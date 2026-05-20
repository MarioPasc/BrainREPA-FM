"""Unit tests for the shared reconstruction-quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from brainrepa_fm.common.metrics import mse, psnr, ssim3d, ssim3d_map
from brainrepa_fm.preflight.augmentation.vae_composability import _psnr

pytestmark = pytest.mark.unit


def _vol(seed: int, shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_mse_identity_is_zero():
    x = _vol(0)
    assert mse(x, x) == pytest.approx(0.0)


def test_mse_known_offset():
    x = np.zeros((8, 8, 8), dtype=np.float32)
    y = np.full((8, 8, 8), 0.5, dtype=np.float32)
    assert mse(x, y) == pytest.approx(0.25)


def test_mse_empty_mask_is_nan():
    x, y = _vol(1), _vol(2)
    assert np.isnan(mse(x, y, np.zeros((16, 16, 16), dtype=bool)))


def test_psnr_identity_is_inf():
    x = _vol(3)
    assert psnr(x, x) == float("inf")


def test_psnr_empty_mask_is_nan():
    x, y = _vol(4), _vol(5)
    assert np.isnan(psnr(x, y, np.zeros((16, 16, 16), dtype=bool)))


def test_psnr_data_range_scales_by_20log10():
    x = np.zeros((8, 8, 8), dtype=np.float32)
    y = np.full((8, 8, 8), 0.1, dtype=np.float32)
    delta = psnr(x, y, data_range=255.0) - psnr(x, y, data_range=1.0)
    assert delta == pytest.approx(20.0 * np.log10(255.0), rel=1e-6)


def test_psnr_matches_private_augmentation_helper():
    x, y = _vol(6), _vol(7)
    assert psnr(x, y) == pytest.approx(_psnr(x, y), rel=1e-9)
    m = np.zeros((16, 16, 16), dtype=bool)
    m[2:8, 2:8, 2:8] = True
    assert psnr(x, y, m) == pytest.approx(_psnr(x, y, m), rel=1e-9)


def test_ssim3d_identity_is_one():
    x = _vol(8)
    assert ssim3d(x, x) == pytest.approx(1.0, abs=1e-6)


def test_ssim3d_degrades_under_noise():
    x = _vol(9)
    noisy = np.clip(
        x + np.random.default_rng(10).normal(0.0, 0.2, x.shape), 0.0, 1.0
    ).astype(np.float32)
    assert ssim3d(x, noisy) < ssim3d(x, x)


def test_ssim3d_empty_mask_is_nan():
    x, y = _vol(11), _vol(12)
    assert np.isnan(ssim3d(x, y, np.zeros((16, 16, 16), dtype=bool)))


def test_ssim3d_map_has_input_shape():
    x, y = _vol(13), _vol(14)
    assert ssim3d_map(x, y).shape == x.shape


def test_ssim3d_map_rejects_non_3d():
    with pytest.raises(ValueError):
        ssim3d_map(np.zeros((4, 4)), np.zeros((4, 4)))
