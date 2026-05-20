"""Unit tests for the pre-flight 03 figures — file is produced and non-empty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brainrepa_fm.preflight.maisi_vae.visualize import (
    render_latent_stats_figure,
    render_psnr_histogram,
    render_reconstruction_montage,
    render_ssim_histogram,
    render_voided_drop_histogram,
    render_voided_roundtrip_montage,
    render_voided_scatter,
)

pytestmark = [pytest.mark.unit, pytest.mark.preflight_maisi]


def _assert_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_psnr_histogram_writes_png(tmp_path):
    out = tmp_path / "psnr.png"
    render_psnr_histogram(
        [28.1, 30.2, 25.0, float("nan")],
        region_label="inside void",
        threshold_db=28.0,
        out_path=out,
    )
    _assert_png(out)


def test_ssim_histogram_writes_png(tmp_path):
    out = tmp_path / "ssim.png"
    render_ssim_histogram([0.90, 0.85, 0.92], region_label="brain", out_path=out)
    _assert_png(out)


def test_reconstruction_montage_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.random((24, 24, 20)).astype(np.float32)
    recon = np.clip(gt + rng.normal(0.0, 0.05, gt.shape), 0.0, 1.0).astype(np.float32)
    void = np.zeros((24, 24, 20), dtype=np.int8)
    void[4:10, 4:10, 4:10] = 1
    out = tmp_path / "montage.png"
    render_reconstruction_montage(
        gt_volume=gt, reconstructed=recon, void_mask=void, label="best", out_path=out
    )
    _assert_png(out)


def test_latent_stats_figure_writes_png(tmp_path):
    out = tmp_path / "latent.png"
    render_latent_stats_figure((0.01, -0.02, 0.0, 0.03), (1.02, 0.98, 1.0, 0.99), out_path=out)
    _assert_png(out)


def test_voided_scatter_writes_png(tmp_path):
    out = tmp_path / "voided.png"
    render_voided_scatter([1.0, 2.0, 1.5], [0.01, 0.02, 0.0], out_path=out)
    _assert_png(out)


def test_voided_drop_histogram_writes_png(tmp_path):
    out = tmp_path / "drop.png"
    render_voided_drop_histogram([0.5, 1.2, -0.1, float("nan")], out_path=out)
    _assert_png(out)


def test_voided_roundtrip_montage_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    gt = rng.random((24, 24, 20)).astype(np.float32)
    void = np.zeros((24, 24, 20), dtype=np.int8)
    void[4:12, 4:12, 4:12] = 1
    voided = gt.copy()
    voided[void.astype(bool)] = 0.0
    recon_voided = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32)
    out = tmp_path / "voided_montage.png"
    render_voided_roundtrip_montage(
        gt_volume=gt,
        voided_volume=voided,
        recon_voided=recon_voided,
        void_mask=void,
        label="worst_drop",
        out_path=out,
    )
    _assert_png(out)


def test_psnr_histogram_handles_all_nan(tmp_path):
    out = tmp_path / "empty.png"
    render_psnr_histogram([float("nan"), float("nan")], region_label="inside void", out_path=out)
    _assert_png(out)
