"""Patient-level partitioning correctness."""

from __future__ import annotations

import numpy as np
import pytest

from brainrepa_fm.data.brats_partition import extract_patient_id, partition_patients


@pytest.mark.unit
def test_extract_patient_id() -> None:
    assert extract_patient_id("BraTS-GLI-00007-001") == "00007"
    with pytest.raises(ValueError):
        extract_patient_id("not-a-bratsid")


@pytest.mark.unit
def test_partition_disjoint_and_exhaustive() -> None:
    # 10 patients, 2 sessions each.
    scan_ids = [f"BraTS-GLI-{p:05d}-{s:03d}" for p in range(10) for s in range(2)]
    parts = partition_patients(scan_ids, fractions=(0.6, 0.2, 0.2), seed=42)
    combined = np.concatenate([parts[k] for k in ("train", "val", "test")])
    assert sorted(combined.tolist()) == sorted(range(len(scan_ids)))
    assert len(set(combined.tolist())) == len(scan_ids)


@pytest.mark.unit
def test_partition_patient_atomicity() -> None:
    """No patient appears in more than one split."""
    scan_ids = [f"BraTS-GLI-{p:05d}-{s:03d}" for p in range(20) for s in range(3)]
    parts = partition_patients(scan_ids, fractions=(0.7, 0.15, 0.15), seed=0)

    def patients_for(split: str) -> set[str]:
        return {extract_patient_id(scan_ids[i]) for i in parts[split].tolist()}

    assert patients_for("train").isdisjoint(patients_for("val"))
    assert patients_for("train").isdisjoint(patients_for("test"))
    assert patients_for("val").isdisjoint(patients_for("test"))


@pytest.mark.unit
def test_partition_seed_deterministic() -> None:
    scan_ids = [f"BraTS-GLI-{p:05d}-000" for p in range(50)]
    a = partition_patients(scan_ids, fractions=(0.8, 0.1, 0.1), seed=7)
    b = partition_patients(scan_ids, fractions=(0.8, 0.1, 0.1), seed=7)
    for k in ("train", "val", "test"):
        np.testing.assert_array_equal(a[k], b[k])


@pytest.mark.unit
def test_partition_bad_fractions_raise() -> None:
    with pytest.raises(ValueError):
        partition_patients(["BraTS-GLI-00000-000"], fractions=(0.4, 0.4, 0.4), seed=0)
