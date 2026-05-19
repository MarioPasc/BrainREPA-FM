"""Smoke test for the BraTS-2026 converter (uses on-disk dataset; marked preflight_aug)."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from brainrepa_fm.data.brats2026_converter import BraTS2026ConvertConfig, BraTS2026Converter
from brainrepa_fm.data.brats2026_schema import assert_brats2026_valid

DATA_TRAIN = Path(
    "/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
)
DATA_VAL = Path(
    "/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
)


@pytest.mark.preflight_aug
@pytest.mark.skipif(
    not (DATA_TRAIN.exists() and DATA_VAL.exists()),
    reason="BraTS-2026 source directories not present",
)
def test_converter_smoke(tmp_path: Path) -> None:
    out = tmp_path / "smoke.h5"
    cfg = BraTS2026ConvertConfig(
        training_root=DATA_TRAIN,
        challenge_val_root=DATA_VAL,
        output_path=out,
        max_subjects_training=2,
        max_subjects_challenge_val=2,
        partition_fractions=(0.5, 0.0, 0.5),
        partition_seed=0,
    )
    produced = BraTS2026Converter(cfg).run()
    assert produced == out
    assert_brats2026_valid(produced)

    with h5py.File(produced, "r") as f:
        assert int(f.attrs["n_scans"]) == 4
        assert int(f["gt/scan_index"].shape[0]) == 2
        assert tuple(f["images/t1_voided"].shape[1:]) == (240, 240, 155)
        assert f["images/t1_voided"][0].min() >= 0.0
        assert f["images/t1_voided"][0].max() <= 1.0 + 1e-3
