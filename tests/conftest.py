"""Shared fixtures for the BrainREPA-FM test suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from brainrepa_fm.data.h5_schemas import BRATS2026_SCHEMA_VERSION, BRATS_VOLUME_SHAPE


@pytest.fixture
def tiny_brats_h5(tmp_path: Path) -> Path:
    """Build a minimal valid Schema-A H5 with 4 scans (2 train + 1 test + 1 challenge_val)."""
    path = tmp_path / "tiny.h5"
    n = 4
    n_with_gt = 3  # train + test
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = BRATS2026_SCHEMA_VERSION
        f.attrs["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.attrs["producer"] = "tests.conftest.tiny_brats_h5"
        f.attrs["config_json"] = json.dumps({"seed": 0, "fixture": "tiny_brats_h5"})
        f.attrs["git_sha"] = "deadbeef"
        f.attrs["orientation"] = "RAS"
        f.attrs["voxel_spacing_mm"] = json.dumps([1.0, 1.0, 1.0])
        f.attrs["preprocessing"] = "synthetic"
        f.attrs["n_scans"] = n

        vlen = h5py.string_dtype(encoding="utf-8")
        scan_id = f.create_dataset("scan_id", shape=(n,), dtype=vlen)
        cohort = f.create_dataset("cohort", shape=(n,), dtype=vlen)
        split = f.create_dataset("split", shape=(n,), dtype=vlen)
        src = f.create_dataset("metadata/source_path", shape=(n,), dtype=vlen)
        clip = f.create_dataset("metadata/voxel_intensity_clip", shape=(n, 2), dtype=np.float32)
        chunk = (1, *BRATS_VOLUME_SHAPE)

        t1 = f.create_dataset(
            "images/t1_voided",
            shape=(n, *BRATS_VOLUME_SHAPE),
            dtype=np.float32,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )
        brain = f.create_dataset(
            "masks/brain",
            shape=(n, *BRATS_VOLUME_SHAPE),
            dtype=np.int8,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )
        void = f.create_dataset(
            "masks/void",
            shape=(n, *BRATS_VOLUME_SHAPE),
            dtype=np.int8,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )

        gt_scan_index = f.create_dataset("gt/scan_index", shape=(n_with_gt,), dtype=np.int32)
        gt_t1 = f.create_dataset(
            "gt/t1",
            shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
            dtype=np.float32,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )
        gt_h = f.create_dataset(
            "gt/healthy_mask",
            shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
            dtype=np.int8,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )
        gt_t = f.create_dataset(
            "gt/tumor_mask",
            shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
            dtype=np.int8,
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
        )

        # populate
        splits = {"train": [0, 1], "test": [2], "challenge_val": [3], "val": []}
        gt_cursor = 0
        for i in range(n):
            scan_id[i] = f"BraTS-GLI-{i:05d}-000"
            cohort[i] = "GLI"
            # find this index's split
            split[i] = next(name for name, idx in splits.items() if i in idx)
            src[i] = f"/tmp/fixture/{i}"
            clip[i] = np.array([0.05, 0.95], dtype=np.float32)
            t1_arr = rng.random(BRATS_VOLUME_SHAPE).astype(np.float32) * 0.6 + 0.2
            t1_arr[:10, :10, :10] = 0.0  # void
            t1[i] = t1_arr
            brain_arr = (t1_arr > 0).astype(np.int8)
            brain[i] = brain_arr
            void_arr = np.zeros_like(brain_arr)
            void_arr[:10, :10, :10] = 1
            void[i] = void_arr
            if i in splits["train"] or i in splits["test"]:
                gt_scan_index[gt_cursor] = i
                gt_t1[gt_cursor] = t1_arr.copy()  # fake "GT" matches input here
                gt_h[gt_cursor] = void_arr.copy()
                gt_t[gt_cursor] = void_arr.copy()
                gt_cursor += 1

        for name, idx in splits.items():
            f.create_dataset(f"splits/{name}", data=np.asarray(idx, dtype=np.int32))

    return path
