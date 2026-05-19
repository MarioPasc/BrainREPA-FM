"""Validator for the BraTS-2026 source H5 (``brats_inpainting_2026.h5``).

Pair of helpers per ``.claude/rules/h5-design-principles.md`` §7:

- :func:`validate_brats2026` returns a list of human-readable violation strings
  (empty list ⇔ valid file).
- :func:`assert_brats2026_valid` raises :class:`~brainrepa_fm.data.exceptions.BratsH5SchemaError`
  with the joined violation list when the file is non-conformant.

The producer (:mod:`brainrepa_fm.data.brats2026_converter`) calls
:func:`assert_brats2026_valid` on its temp output before atomically moving it
into place. A non-conformant artifact never reaches disk in a "successful"
state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from brainrepa_fm.data.exceptions import BratsH5SchemaError
from brainrepa_fm.data.h5_schemas import (
    ALLOWED_COHORTS,
    ALLOWED_SPLITS,
    BRATS2026_SCHEMA,
    BRATS_VOLUME_SHAPE,
    AttrSpec,
    DatasetSpec,
)

_NUMPY_DTYPES: dict[str, type] = {
    "float32": np.float32,
    "int8": np.int8,
    "int32": np.int32,
}


def _check_attr(value: Any, spec: AttrSpec) -> str | None:
    """Return a violation string if the attr value does not match its spec, else None."""
    if spec.dtype == "str":
        if not isinstance(value, str | bytes):
            return f"root attr '{spec.name}' must be str, got {type(value).__name__}"
    elif spec.dtype == "bool":
        # numpy.bool_, Python bool, or 0/1 integer all acceptable
        try:
            bool(value)
        except (TypeError, ValueError):
            return f"root attr '{spec.name}' must be bool-coercible"
    elif spec.dtype == "float":
        try:
            float(value)
        except (TypeError, ValueError):
            return f"root attr '{spec.name}' must be float-coercible"
    elif spec.dtype == "int":
        try:
            int(value)
        except (TypeError, ValueError):
            return f"root attr '{spec.name}' must be int-coercible"
    return None


def _check_dataset_shape(
    dset: h5py.Dataset, spec: DatasetSpec, n_scans: int, n_with_gt: int
) -> list[str]:
    """Cross-check a dataset's shape against its DatasetSpec."""
    violations: list[str] = []
    expected_leading = {
        "n_scans": n_scans,
        "n_with_gt": n_with_gt,
        # split-index datasets carry their own leading dim — only an upper bound is checked.
        "n_train": n_scans,
        "n_val": n_scans,
        "n_test": n_scans,
        "n_challenge_val": n_scans,
        "n_scans_plus_one": n_scans + 1,
    }.get(spec.leading_dim)

    if spec.trailing_shape is None:
        # 1-D dataset (or split index).
        if dset.ndim != 1:
            violations.append(f"{spec.path}: expected 1-D, got shape {dset.shape}")
        if expected_leading is not None and spec.leading_dim.startswith("n_"):
            if spec.leading_dim in {"n_train", "n_val", "n_test", "n_challenge_val"}:
                if dset.shape[0] > n_scans:
                    violations.append(
                        f"{spec.path}: leading dim {dset.shape[0]} exceeds n_scans={n_scans}"
                    )
            elif dset.shape[0] != expected_leading:
                violations.append(
                    f"{spec.path}: expected leading dim {expected_leading} ({spec.leading_dim}), "
                    f"got {dset.shape[0]}"
                )
    else:
        if dset.ndim != 1 + len(spec.trailing_shape):
            violations.append(
                f"{spec.path}: expected ndim {1 + len(spec.trailing_shape)}, got {dset.ndim}"
            )
            return violations
        if expected_leading is not None and dset.shape[0] != expected_leading:
            violations.append(
                f"{spec.path}: expected leading dim {expected_leading} ({spec.leading_dim}), "
                f"got {dset.shape[0]}"
            )
        if tuple(dset.shape[1:]) != tuple(spec.trailing_shape):
            violations.append(
                f"{spec.path}: expected trailing shape {spec.trailing_shape}, got {tuple(dset.shape[1:])}"
            )
    return violations


def _check_dataset_dtype(dset: h5py.Dataset, spec: DatasetSpec) -> str | None:
    """Cross-check the dataset's dtype string."""
    if spec.dtype == "vlen-str":
        # h5py represents vlen-str as object dtype with special_dtype(vlen=str)
        if h5py.check_string_dtype(dset.dtype) is None:
            return f"{spec.path}: expected vlen-str, got dtype {dset.dtype}"
    elif spec.dtype in _NUMPY_DTYPES:
        expected = np.dtype(_NUMPY_DTYPES[spec.dtype])
        if dset.dtype != expected:
            return f"{spec.path}: expected dtype {expected}, got {dset.dtype}"
    return None


def _check_splits_partition(f: h5py.File, n_scans: int) -> list[str]:
    """Check that splits/{train,val,test,challenge_val} indices form a partition of [0, n_scans)."""
    violations: list[str] = []
    pieces: list[np.ndarray] = []
    for name in ALLOWED_SPLITS:
        path = f"splits/{name}"
        if path not in f:
            violations.append(f"{path}: required split is missing")
            continue
        idx = f[path][...]
        if idx.size == 0:
            continue
        if idx.dtype != np.int32:
            violations.append(f"{path}: expected int32 indices, got {idx.dtype}")
        if idx.min() < 0 or idx.max() >= n_scans:
            violations.append(
                f"{path}: indices must be in [0, {n_scans}); got [{idx.min()}, {idx.max()}]"
            )
        if len(set(idx.tolist())) != idx.size:
            violations.append(f"{path}: indices must be unique within a split")
        pieces.append(idx)

    if pieces:
        combined = np.concatenate(pieces)
        if combined.size != n_scans:
            violations.append(
                f"splits/* must partition range({n_scans}) exhaustively; got {combined.size} indices"
            )
        elif len(set(combined.tolist())) != n_scans:
            violations.append("splits/* indices overlap across splits")
        elif set(combined.tolist()) != set(range(n_scans)):
            violations.append("splits/* indices do not cover [0, n_scans) exactly")
    return violations


def _check_gt_consistency(f: h5py.File, n_scans: int) -> list[str]:
    """The ``gt/`` group is sparse: scan_index drives the leading dim of every gt/ dataset.

    All gt/ datasets must share the same leading dim. Indices must be unique,
    in-range, and disjoint from ``splits/challenge_val`` (challenge-val has no GT).
    """
    violations: list[str] = []
    required = ("gt/scan_index", "gt/t1", "gt/healthy_mask", "gt/tumor_mask")
    present = [p for p in required if p in f]
    if not present:
        return violations  # entire group absent is allowed (e.g. test fixtures)
    if len(present) != len(required):
        missing = [p for p in required if p not in present]
        violations.append(f"gt/* partial: missing {missing} (all four are required together)")
        return violations

    scan_index = f["gt/scan_index"][...]
    n_gt = scan_index.shape[0]
    for path in ("gt/t1", "gt/healthy_mask", "gt/tumor_mask"):
        if f[path].shape[0] != n_gt:
            violations.append(
                f"{path}.shape[0] = {f[path].shape[0]} disagrees with gt/scan_index.shape[0] = {n_gt}"
            )

    if n_gt > 0:
        if scan_index.min() < 0 or scan_index.max() >= n_scans:
            violations.append(
                f"gt/scan_index: indices must be in [0, {n_scans}); "
                f"got [{scan_index.min()}, {scan_index.max()}]"
            )
        if len(set(scan_index.tolist())) != n_gt:
            violations.append("gt/scan_index: indices must be unique")

    if "splits/challenge_val" in f and n_gt > 0:
        chal = set(f["splits/challenge_val"][...].tolist())
        leaks = sorted(set(scan_index.tolist()) & chal)
        if leaks:
            violations.append(
                f"gt/scan_index: leaks {len(leaks)} challenge_val indices "
                f"(first few: {leaks[:5]}) — challenge_val rows have no GT by construction"
            )

    # gt/t1 first-row value range.
    if n_gt > 0 and "gt/t1" in f:
        sample = f["gt/t1"][0]
        if sample.min() < -1e-3 or sample.max() > 1 + 1e-3:
            violations.append(
                f"gt/t1: first row outside [0, 1]: "
                f"min={float(sample.min()):.4f} max={float(sample.max()):.4f}"
            )

    return violations


def _check_split_values(f: h5py.File) -> list[str]:
    """``split`` dataset values must be in ALLOWED_SPLITS."""
    if "split" not in f:
        return []
    raw = f["split"][...]
    decoded = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
    bad = sorted({s for s in decoded if s not in ALLOWED_SPLITS})
    if bad:
        return [f"split: contains values outside ALLOWED_SPLITS={ALLOWED_SPLITS}: {bad}"]
    return []


def _check_cohort_values(f: h5py.File) -> list[str]:
    if "cohort" not in f:
        return []
    raw = f["cohort"][...]
    decoded = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
    bad = sorted({s for s in decoded if s not in ALLOWED_COHORTS})
    if bad:
        return [f"cohort: contains values outside ALLOWED_COHORTS={ALLOWED_COHORTS}: {bad}"]
    return []


def _check_config_json(f: h5py.File) -> list[str]:
    """``config_json`` root attr must be parseable JSON."""
    if "config_json" not in f.attrs:
        return ["root attr 'config_json' missing"]
    try:
        json.loads(f.attrs["config_json"])
    except (TypeError, ValueError) as exc:
        return [f"root attr 'config_json' is not valid JSON: {exc}"]
    return []


def validate_brats2026(path: str | Path) -> list[str]:
    """Validate a candidate ``brats_inpainting_2026.h5`` file.

    Parameters:
        path: Filesystem path to the H5 to validate.

    Returns:
        A list of violation strings. An empty list means the file conforms to
        :data:`brainrepa_fm.data.h5_schemas.BRATS2026_SCHEMA`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    violations: list[str] = []

    with h5py.File(path, "r") as f:
        # 1. Root attrs
        for spec in BRATS2026_SCHEMA.root_attrs:
            if spec.name not in f.attrs:
                if spec.required:
                    violations.append(f"root attr '{spec.name}' missing")
                continue
            err = _check_attr(f.attrs[spec.name], spec)
            if err:
                violations.append(err)

        # Schema-version pin
        if "schema_version" in f.attrs:
            actual_version = f.attrs["schema_version"]
            if isinstance(actual_version, bytes):
                actual_version = actual_version.decode()
            if str(actual_version) != BRATS2026_SCHEMA.schema_version:
                violations.append(
                    f"schema_version mismatch: file has '{actual_version}', "
                    f"validator supports '{BRATS2026_SCHEMA.schema_version}'"
                )

        violations.extend(_check_config_json(f))

        # 2. n_scans (derived from scan_id)
        if "scan_id" not in f:
            violations.append("required dataset 'scan_id' missing — cannot infer n_scans")
            return violations
        n_scans = int(f["scan_id"].shape[0])
        if "n_scans" in f.attrs and int(f.attrs["n_scans"]) != n_scans:
            violations.append(
                f"root attr 'n_scans' = {int(f.attrs['n_scans'])} disagrees with scan_id.shape[0] = {n_scans}"
            )

        # 3. Required per-scan datasets, shapes and dtypes
        n_with_gt = int(f["gt/scan_index"].shape[0]) if "gt/scan_index" in f else 0
        for spec in BRATS2026_SCHEMA.datasets:
            if spec.path not in f:
                if spec.required:
                    violations.append(f"required dataset '{spec.path}' missing")
                continue
            dset = f[spec.path]
            violations.extend(_check_dataset_shape(dset, spec, n_scans, n_with_gt))
            err = _check_dataset_dtype(dset, spec)
            if err:
                violations.append(err)

        # 4. Cross-field invariants
        # 4a. T1_voided / brain / void volume shapes; cross-check they share n_scans.
        for required in ("images/t1_voided", "masks/brain", "masks/void"):
            if required in f and f[required].shape[0] != n_scans:
                violations.append(
                    f"{required}.shape[0] = {f[required].shape[0]} disagrees with n_scans = {n_scans}"
                )
            if required in f and tuple(f[required].shape[1:]) != BRATS_VOLUME_SHAPE:
                violations.append(
                    f"{required}: per-scan shape {f[required].shape[1:]} != BRATS_VOLUME_SHAPE {BRATS_VOLUME_SHAPE}"
                )

        # 4b. masks/brain and masks/void value range
        for binary in ("masks/brain", "masks/void"):
            if binary in f and n_scans > 0:
                sample = f[binary][0]
                vals = np.unique(sample)
                if not set(vals.tolist()).issubset({0, 1}):
                    violations.append(
                        f"{binary}: values outside {{0,1}} in first scan ({vals.tolist()})"
                    )

        # 4c. images/t1_voided value range — first scan should be in [0, 1] post-normalization.
        if "images/t1_voided" in f and n_scans > 0:
            sample = f["images/t1_voided"][0]
            if sample.min() < -1e-3 or sample.max() > 1 + 1e-3:
                violations.append(
                    f"images/t1_voided: first scan outside [0, 1] post-normalization: "
                    f"min={float(sample.min()):.4f} max={float(sample.max()):.4f}"
                )

        # 5. Splits partition
        violations.extend(_check_splits_partition(f, n_scans))

        # 6. Categorical values
        violations.extend(_check_split_values(f))
        violations.extend(_check_cohort_values(f))

        # 7. GT consistency
        violations.extend(_check_gt_consistency(f, n_scans))

    return violations


def assert_brats2026_valid(path: str | Path) -> None:
    """Raise :class:`BratsH5SchemaError` if the file does not conform to Schema A.

    Parameters:
        path: Filesystem path to the H5 to validate.

    Raises:
        BratsH5SchemaError: With a joined violation list as the message.
        FileNotFoundError: If ``path`` does not exist.
    """
    violations = validate_brats2026(path)
    if violations:
        msg = f"{Path(path)} does not conform to BRATS2026_SCHEMA:\n  - " + "\n  - ".join(
            violations
        )
        raise BratsH5SchemaError(msg)


__all__ = ["assert_brats2026_valid", "validate_brats2026"]
