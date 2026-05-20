"""Unit tests for the MAISI-envelope fit utility."""

from __future__ import annotations

import numpy as np
import pytest

from brainrepa_fm.preflight.maisi_vae.preprocess import prepare_to_envelope

pytestmark = pytest.mark.unit


def test_crop_brats_to_3060_envelope():
    vol = np.random.default_rng(0).random((240, 240, 155)).astype(np.float32)
    assert prepare_to_envelope(vol, (192, 192, 144)).shape == (192, 192, 144)


def test_pad_brats_to_a100_envelope():
    vol = np.random.default_rng(1).random((240, 240, 155)).astype(np.float32)
    assert prepare_to_envelope(vol, (256, 256, 192)).shape == (256, 256, 192)


def test_pad_region_carries_no_signal():
    vol = np.ones((10, 10, 10), dtype=np.float32)
    out = prepare_to_envelope(vol, (16, 16, 16))
    assert out.shape == (16, 16, 16)
    assert out.sum() == pytest.approx(1000.0)  # 10**3 ones, padded with zeros


def test_center_crop_keeps_the_centre_voxel():
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[10, 10, 10] = 1.0
    out = prepare_to_envelope(vol, (8, 8, 8))
    assert out.shape == (8, 8, 8)
    assert out.sum() == pytest.approx(1.0)


def test_mixed_crop_and_pad_per_axis():
    vol = np.ones((20, 4, 12), dtype=np.float32)  # crop, pad, crop
    assert prepare_to_envelope(vol, (8, 8, 8)).shape == (8, 8, 8)


def test_mask_dtype_is_preserved():
    mask = np.ones((10, 10, 10), dtype=np.int8)
    assert prepare_to_envelope(mask, (16, 16, 16)).dtype == np.int8


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        prepare_to_envelope(np.zeros((4, 4)), (8, 8, 8))
