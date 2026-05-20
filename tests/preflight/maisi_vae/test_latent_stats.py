"""Unit tests for the streaming per-channel latent statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brainrepa_fm.preflight.maisi_vae.latent_stats import (
    LatentChannelStats,
    LatentStatsAccumulator,
)

pytestmark = [pytest.mark.unit, pytest.mark.preflight_maisi]


def test_single_volume_matches_numpy():
    z = np.random.default_rng(0).normal(0.3, 1.2, size=(1, 4, 6, 6, 6)).astype(np.float32)
    acc = LatentStatsAccumulator(4)
    acc.update(torch.from_numpy(z))
    res = acc.result()
    per_channel = np.moveaxis(z, 1, 0).reshape(4, -1).astype(np.float64)
    np.testing.assert_allclose(res.mean, per_channel.mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(res.std, per_channel.std(axis=1), rtol=1e-5)


def test_streaming_matches_pooled_estimate():
    rng = np.random.default_rng(1)
    z1 = rng.normal(0.0, 1.0, size=(1, 4, 5, 5, 5)).astype(np.float32)
    z2 = rng.normal(2.0, 3.0, size=(1, 4, 7, 7, 7)).astype(np.float32)
    acc = LatentStatsAccumulator(4)
    acc.update(torch.from_numpy(z1))
    acc.update(torch.from_numpy(z2))
    res = acc.result()
    pooled = np.concatenate(
        [np.moveaxis(z1, 1, 0).reshape(4, -1), np.moveaxis(z2, 1, 0).reshape(4, -1)], axis=1
    ).astype(np.float64)
    np.testing.assert_allclose(res.mean, pooled.mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(res.std, pooled.std(axis=1), rtol=1e-5)


def test_result_reports_four_channels():
    acc = LatentStatsAccumulator(4)
    acc.update(np.zeros((1, 4, 3, 3, 3), dtype=np.float32))
    res = acc.result()
    assert isinstance(res, LatentChannelStats)
    assert len(res.mean) == 4
    assert len(res.std) == 4


def test_channel_count_mismatch_raises():
    acc = LatentStatsAccumulator(4)
    with pytest.raises(ValueError):
        acc.update(np.zeros((1, 3, 3, 3, 3), dtype=np.float32))


def test_rejects_non_5d_latent():
    acc = LatentStatsAccumulator(4)
    with pytest.raises(ValueError):
        acc.update(np.zeros((4, 3, 3, 3), dtype=np.float32))
