"""Mutation tests for Schema B (forward-declared latent H5)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from brainrepa_fm.data.brainrepa_latents_schema import (
    assert_brainrepa_latents_valid,
    validate_brainrepa_latents,
)
from brainrepa_fm.data.exceptions import LatentsH5SchemaError
from brainrepa_fm.data.h5_schemas import LATENTS_SCHEMA_VERSION


def _build_tiny_latents(path: Path) -> None:
    n = 3
    c, lz, ly, lx = 4, 48, 48, 36
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = LATENTS_SCHEMA_VERSION
        f.attrs["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.attrs["producer"] = "tests.fixture"
        f.attrs["config_json"] = json.dumps({"seed": 0})
        f.attrs["git_sha"] = "deadbeef"
        f.attrs["n_scans"] = n
        f.attrs["latent_stats_calibrated"] = False
        f.attrs["vae_checkpoint_sha256"] = "b5ed556dc64872ca"
        f.attrs["vae_scale_factor"] = 1.0
        f.attrs["latent_channels"] = c
        f.attrs["latent_spatial_shape"] = json.dumps([lz, ly, lx])

        vlen = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "scan_id",
            data=np.array([f"BraTS-GLI-{i:05d}-000" for i in range(n)], dtype=object),
            dtype=vlen,
        )
        f.create_dataset(
            "split", data=np.array(["train", "val", "challenge_val"], dtype=object), dtype=vlen
        )
        f.create_dataset("latents/anchor", data=np.zeros((n, c, lz, ly, lx), dtype=np.float32))
        # CSR: 1 view per scan (just for testing).
        f.create_dataset(
            "latents/augmented/values", data=np.zeros((n, c, lz, ly, lx), dtype=np.float32)
        )
        f.create_dataset("latents/augmented/offsets", data=np.array([0, 1, 2, 3], dtype=np.int32))
        f.create_dataset(
            "latents/augmented/augmentation_ids",
            data=np.array(["A.1", "A.1", "A.1"], dtype=object),
            dtype=vlen,
        )
        f.create_dataset("augmentations/include", data=np.array(["A.1"], dtype=object), dtype=vlen)
        f.create_dataset("latent_scale", data=np.zeros(c, dtype=np.float32))
        f.create_dataset("latent_mean", data=np.zeros(c, dtype=np.float32))
        f.create_dataset("splits/train", data=np.array([0], dtype=np.int32))
        f.create_dataset("splits/val", data=np.array([1], dtype=np.int32))
        f.create_dataset("splits/test", data=np.array([], dtype=np.int32))
        f.create_dataset("splits/challenge_val", data=np.array([2], dtype=np.int32))


@pytest.mark.unit
def test_tiny_latents_validates(tmp_path: Path) -> None:
    p = tmp_path / "tiny.h5"
    _build_tiny_latents(p)
    assert validate_brainrepa_latents(p) == []


@pytest.mark.unit
def test_csr_offsets_non_monotonic_raises(tmp_path: Path) -> None:
    p = tmp_path / "tiny.h5"
    _build_tiny_latents(p)
    with h5py.File(p, "a") as f:
        del f["latents/augmented/offsets"]
        f.create_dataset("latents/augmented/offsets", data=np.array([0, 2, 1, 3], dtype=np.int32))
    with pytest.raises(LatentsH5SchemaError, match="monotonically"):
        assert_brainrepa_latents_valid(p)


@pytest.mark.unit
def test_aug_id_not_in_include_raises(tmp_path: Path) -> None:
    p = tmp_path / "tiny.h5"
    _build_tiny_latents(p)
    vlen = h5py.string_dtype(encoding="utf-8")
    with h5py.File(p, "a") as f:
        del f["latents/augmented/augmentation_ids"]
        f.create_dataset(
            "latents/augmented/augmentation_ids",
            data=np.array(["A.1", "ROGUE", "A.1"], dtype=object),
            dtype=vlen,
        )
    with pytest.raises(LatentsH5SchemaError, match="ROGUE"):
        assert_brainrepa_latents_valid(p)


@pytest.mark.unit
def test_latent_spatial_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "tiny.h5"
    _build_tiny_latents(p)
    with h5py.File(p, "a") as f:
        f.attrs["latent_spatial_shape"] = json.dumps([99, 99, 99])
    with pytest.raises(LatentsH5SchemaError, match="latent_spatial_shape"):
        assert_brainrepa_latents_valid(p)
