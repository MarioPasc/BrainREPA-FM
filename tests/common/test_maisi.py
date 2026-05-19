"""MAISI VAE wrapper smoke (GPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_maisi_round_trip_3060_envelope() -> None:
    from brainrepa_fm.common.maisi import (
        MAISI_PAD_SHAPE_3060,
        MaisiVAE,
        center_crop_to_maisi,
        probe_latent_shape,
        tensor_to_volume,
        volume_to_tensor,
    )

    vae = MaisiVAE(device="cuda", autocast_fp16=True, use_checkpointing=True)
    assert vae.info.sha256_prefix == "b5ed556dc64872ca"

    # Latent shape at the 3060 envelope.
    shape = probe_latent_shape(vae, input_shape=MAISI_PAD_SHAPE_3060)
    assert shape == (4, 48, 48, 36)

    # Round-trip a brain-like input.
    rng = np.random.RandomState(0)
    x = rng.rand(240, 240, 155).astype(np.float32) * 0.6 + 0.2
    t = volume_to_tensor(x, device="cuda")
    t_crop, _ = center_crop_to_maisi(t, target_shape=MAISI_PAD_SHAPE_3060)
    y = vae.encode_decode(t_crop)
    assert tuple(y.shape) == (1, 1, *MAISI_PAD_SHAPE_3060)
    arr = tensor_to_volume(y.float())
    assert not np.isnan(arr).any()
