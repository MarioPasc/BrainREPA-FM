"""GPU smoke tests for pre-flight 03 — require CUDA and the real MAISI checkpoint.

Skipped by the default ``-m "not gpu and not slow"`` selection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brainrepa_fm.common.maisi import DEFAULT_MAISI_CHECKPOINT

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.preflight_maisi]

_HAVE_ENV = torch.cuda.is_available() and DEFAULT_MAISI_CHECKPOINT.exists()


@pytest.mark.skipif(not _HAVE_ENV, reason="CUDA and the MAISI checkpoint are required")
def test_real_vae_round_trip_is_finite():
    from brainrepa_fm.common.maisi import (
        MAISI_PAD_SHAPE_3060,
        MaisiVAE,
        probe_latent_shape,
        tensor_to_volume,
        volume_to_tensor,
    )
    from brainrepa_fm.preflight.maisi_vae.reconstruction import compute_reconstruction_metrics

    vae = MaisiVAE(device="cuda")
    latent_shape = probe_latent_shape(vae, input_shape=MAISI_PAD_SHAPE_3060, device="cuda")
    assert latent_shape[0] == 4

    rng = np.random.default_rng(0)
    gt = (rng.random(MAISI_PAD_SHAPE_3060).astype(np.float32) * 0.6 + 0.2)
    recon = tensor_to_volume(
        vae.encode_decode(volume_to_tensor(gt, device="cuda")).float()
    )
    brain = np.ones(MAISI_PAD_SHAPE_3060, dtype=np.int8)
    void = np.zeros(MAISI_PAD_SHAPE_3060, dtype=np.int8)
    void[20:60, 20:60, 20:60] = 1
    res = compute_reconstruction_metrics(
        subject_id="probe",
        gt=gt,
        recon=recon,
        brain_mask=brain,
        tumor_mask=void,
        void_masks=[void],
    )
    assert np.isfinite(res.psnr_full)
    assert res.psnr_full > 0.0
    assert 0.0 <= res.ssim_full <= 1.0
