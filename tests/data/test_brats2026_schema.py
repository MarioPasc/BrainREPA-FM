"""Mutation tests for Schema A — the validator must reject corrupted files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from brainrepa_fm.data.brats2026_schema import (
    assert_brats2026_valid,
    validate_brats2026,
)
from brainrepa_fm.data.exceptions import BratsH5SchemaError


@pytest.mark.unit
def test_tiny_fixture_validates(tiny_brats_h5: Path) -> None:
    """The conftest fixture conforms to BRATS2026_SCHEMA."""
    assert validate_brats2026(tiny_brats_h5) == []
    assert_brats2026_valid(tiny_brats_h5)  # raises iff violations


@pytest.mark.unit
def test_missing_schema_version_attr_raises(tiny_brats_h5: Path) -> None:
    with h5py.File(tiny_brats_h5, "a") as f:
        del f.attrs["schema_version"]
    with pytest.raises(BratsH5SchemaError, match="schema_version"):
        assert_brats2026_valid(tiny_brats_h5)


@pytest.mark.unit
def test_wrong_schema_version_raises(tiny_brats_h5: Path) -> None:
    with h5py.File(tiny_brats_h5, "a") as f:
        f.attrs["schema_version"] = "9.9"
    with pytest.raises(BratsH5SchemaError, match="schema_version mismatch"):
        assert_brats2026_valid(tiny_brats_h5)


@pytest.mark.unit
def test_bad_config_json_raises(tiny_brats_h5: Path) -> None:
    with h5py.File(tiny_brats_h5, "a") as f:
        f.attrs["config_json"] = "not json {{{"
    with pytest.raises(BratsH5SchemaError, match="config_json"):
        assert_brats2026_valid(tiny_brats_h5)


@pytest.mark.unit
def test_t1_voided_out_of_range_raises(tiny_brats_h5: Path) -> None:
    with h5py.File(tiny_brats_h5, "a") as f:
        arr = f["images/t1_voided"][0]
        arr[0, 0, 0] = 5.0
        f["images/t1_voided"][0] = arr
    with pytest.raises(BratsH5SchemaError, match=r"images/t1_voided.*\[0, 1\]"):
        assert_brats2026_valid(tiny_brats_h5)


@pytest.mark.unit
def test_split_partition_overlap_raises(tiny_brats_h5: Path) -> None:
    """If a scan appears in two splits, validator must flag the partition violation."""
    with h5py.File(tiny_brats_h5, "a") as f:
        # Move scan 0 into both train and val
        del f["splits/val"]
        f.create_dataset("splits/val", data=np.asarray([0], dtype=np.int32))
    violations = validate_brats2026(tiny_brats_h5)
    assert any("partition" in v or "overlap" in v or "exhaustively" in v for v in violations), (
        violations
    )


@pytest.mark.unit
def test_gt_leak_into_challenge_val_raises(tiny_brats_h5: Path) -> None:
    """gt/scan_index must be disjoint from splits/challenge_val."""
    with h5py.File(tiny_brats_h5, "a") as f:
        old = f["gt/scan_index"][...]
        del f["gt/scan_index"]
        # Add a challenge_val index (3) to GT — should fail.
        f.create_dataset("gt/scan_index", data=np.append(old, np.int32(3)).astype(np.int32))
        # We do NOT extend the values arrays, so we also expect a length-mismatch violation;
        # both should be flagged.
    violations = validate_brats2026(tiny_brats_h5)
    assert any("challenge_val" in v for v in violations), violations
