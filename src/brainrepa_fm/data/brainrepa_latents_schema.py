"""Validator for the latent H5 (``brainrepa_latents.h5``).

The producer for this H5 is a downstream task (post pre-flight 01 decision).
The schema is forward-declared and validated here so that future producers and
consumers share a single contract.

Pair of helpers per ``.claude/rules/h5-design-principles.md`` §7:

- :func:`validate_brainrepa_latents` returns a list of violation strings.
- :func:`assert_brainrepa_latents_valid` raises
  :class:`~brainrepa_fm.data.exceptions.LatentsH5SchemaError`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from brainrepa_fm.data.exceptions import LatentsH5SchemaError
from brainrepa_fm.data.h5_schemas import (
    ALLOWED_SPLITS,
    LATENTS_SCHEMA,
    AttrSpec,
)


def _check_attr(value: Any, spec: AttrSpec) -> str | None:
    """Cross-check one root attr value against its spec."""
    if spec.dtype == "str":
        if not isinstance(value, str | bytes):
            return f"root attr '{spec.name}' must be str, got {type(value).__name__}"
    elif spec.dtype == "bool":
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


def _check_csr_invariants(f: h5py.File, n_scans: int) -> list[str]:
    """Check CSR layout of latents/augmented/{offsets,values,augmentation_ids}."""
    violations: list[str] = []
    if "latents/augmented/offsets" not in f:
        return violations  # absence is allowed if no augmentations are stored yet

    offsets = f["latents/augmented/offsets"][...]
    if offsets.dtype != np.int32:
        violations.append(f"latents/augmented/offsets: expected int32, got {offsets.dtype}")
    if offsets.shape != (n_scans + 1,):
        violations.append(
            f"latents/augmented/offsets: expected shape ({n_scans + 1},), got {offsets.shape}"
        )
        return violations
    if offsets[0] != 0:
        violations.append(f"latents/augmented/offsets[0] must be 0; got {offsets[0]}")
    if np.any(np.diff(offsets) < 0):
        violations.append("latents/augmented/offsets must be monotonically non-decreasing")

    n_aug_rows_offsets = int(offsets[-1])
    if "latents/augmented/values" in f:
        actual_rows = int(f["latents/augmented/values"].shape[0])
        if actual_rows != n_aug_rows_offsets:
            violations.append(
                f"latents/augmented/values has {actual_rows} rows but offsets imply {n_aug_rows_offsets}"
            )
    if "latents/augmented/augmentation_ids" in f:
        actual_rows = int(f["latents/augmented/augmentation_ids"].shape[0])
        if actual_rows != n_aug_rows_offsets:
            violations.append(
                f"latents/augmented/augmentation_ids has {actual_rows} rows but offsets imply {n_aug_rows_offsets}"
            )
    return violations


def _check_augmentation_id_membership(f: h5py.File) -> list[str]:
    """Every augmentation_id in CSR rows must appear in augmentations/include."""
    if "augmentations/include" not in f or "latents/augmented/augmentation_ids" not in f:
        return []
    include_raw = f["augmentations/include"][...]
    rows_raw = f["latents/augmented/augmentation_ids"][...]
    include = {s.decode() if isinstance(s, bytes) else str(s) for s in include_raw}
    rows = {s.decode() if isinstance(s, bytes) else str(s) for s in rows_raw}
    extra = sorted(rows - include)
    if extra:
        return [
            f"latents/augmented/augmentation_ids: contains IDs not in augmentations/include: {extra}"
        ]
    return []


def _check_split_partition(f: h5py.File, n_scans: int) -> list[str]:
    """Same partition rule as Schema A."""
    violations: list[str] = []
    pieces: list[np.ndarray] = []
    for name in ALLOWED_SPLITS:
        path = f"splits/{name}"
        if path not in f:
            violations.append(f"{path}: required split is missing")
            continue
        idx = f[path][...]
        if idx.dtype != np.int32:
            violations.append(f"{path}: expected int32, got {idx.dtype}")
        if idx.size > 0 and (idx.min() < 0 or idx.max() >= n_scans):
            violations.append(
                f"{path}: indices must be in [0, {n_scans}); got [{idx.min()}, {idx.max()}]"
            )
        if len(set(idx.tolist())) != idx.size:
            violations.append(f"{path}: indices must be unique")
        pieces.append(idx)
    if pieces:
        combined = np.concatenate(pieces)
        if combined.size != n_scans or set(combined.tolist()) != set(range(n_scans)):
            violations.append("splits/* must partition range(n_scans) exhaustively without overlap")
    return violations


def _check_latent_stats_shapes(f: h5py.File) -> list[str]:
    violations: list[str] = []
    n_channels = int(f.attrs.get("latent_channels", 4))
    for name in ("latent_scale", "latent_mean"):
        if name not in f:
            violations.append(f"required root dataset '{name}' missing")
            continue
        if f[name].shape != (n_channels,):
            violations.append(f"{name}: expected shape ({n_channels},), got {f[name].shape}")
        if f[name].dtype != np.float32:
            violations.append(f"{name}: expected float32, got {f[name].dtype}")
    return violations


def validate_brainrepa_latents(path: str | Path) -> list[str]:
    """Validate a candidate ``brainrepa_latents.h5`` file.

    Parameters:
        path: Filesystem path to the H5.

    Returns:
        Violation strings (empty list ⇔ valid).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    violations: list[str] = []

    with h5py.File(path, "r") as f:
        # 1. Root attrs
        for spec in LATENTS_SCHEMA.root_attrs:
            if spec.name not in f.attrs:
                if spec.required:
                    violations.append(f"root attr '{spec.name}' missing")
                continue
            err = _check_attr(f.attrs[spec.name], spec)
            if err:
                violations.append(err)

        if "schema_version" in f.attrs:
            v = f.attrs["schema_version"]
            if isinstance(v, bytes):
                v = v.decode()
            if str(v) != LATENTS_SCHEMA.schema_version:
                violations.append(
                    f"schema_version mismatch: file has '{v}', validator supports "
                    f"'{LATENTS_SCHEMA.schema_version}'"
                )

        if "config_json" in f.attrs:
            try:
                json.loads(f.attrs["config_json"])
            except (TypeError, ValueError) as exc:
                violations.append(f"root attr 'config_json' is not valid JSON: {exc}")

        if "latent_spatial_shape" in f.attrs:
            try:
                shape = json.loads(f.attrs["latent_spatial_shape"])
                if not (
                    isinstance(shape, list)
                    and len(shape) == 3
                    and all(isinstance(x, int) for x in shape)
                ):
                    violations.append(
                        f"root attr 'latent_spatial_shape' must be JSON [Lz, Ly, Lx]; got {shape!r}"
                    )
            except (TypeError, ValueError) as exc:
                violations.append(f"root attr 'latent_spatial_shape' is not valid JSON: {exc}")

        # 2. n_scans (derived from scan_id)
        if "scan_id" not in f:
            violations.append("required dataset 'scan_id' missing")
            return violations
        n_scans = int(f["scan_id"].shape[0])

        # 3. Two anchor latents: voided (n_scans) and gt (n_with_gt sparse).
        expected_spatial: tuple[int, ...] | None = None
        if "latent_spatial_shape" in f.attrs:
            try:
                shape_list = json.loads(f.attrs["latent_spatial_shape"])
                expected_spatial = tuple(shape_list)
            except (TypeError, ValueError):
                expected_spatial = None

        for name, expected_lead in (
            ("latents/voided_anchor", n_scans),
            ("latents/gt_anchor", None),  # gt_anchor leading dim is n_with_gt, checked separately
        ):
            if name not in f:
                violations.append(f"required dataset '{name}' missing")
                continue
            anchor = f[name]
            if anchor.ndim != 5:
                violations.append(
                    f"{name}: expected 5-D (N, C, Lz, Ly, Lx), got shape {anchor.shape}"
                )
                continue
            if expected_lead is not None and anchor.shape[0] != expected_lead:
                violations.append(
                    f"{name}.shape[0] = {anchor.shape[0]} disagrees with n_scans = {expected_lead}"
                )
            if expected_spatial is not None and tuple(anchor.shape[2:]) != expected_spatial:
                violations.append(
                    f"{name} trailing spatial shape {tuple(anchor.shape[2:])} "
                    f"!= 'latent_spatial_shape' attr {expected_spatial}"
                )

        # gt_anchor must be paired with gt/scan_index of matching length.
        if "latents/gt_anchor" in f:
            n_gt = int(f["latents/gt_anchor"].shape[0])
            if "gt/scan_index" not in f:
                violations.append(
                    "required dataset 'gt/scan_index' missing (paired with latents/gt_anchor)"
                )
            else:
                scan_index = f["gt/scan_index"][...]
                if scan_index.dtype != np.int32:
                    violations.append(f"gt/scan_index: expected int32, got {scan_index.dtype}")
                if scan_index.shape[0] != n_gt:
                    violations.append(
                        f"gt/scan_index.shape[0] = {scan_index.shape[0]} disagrees with "
                        f"latents/gt_anchor.shape[0] = {n_gt}"
                    )
                if n_gt > 0:
                    if scan_index.min() < 0 or scan_index.max() >= n_scans:
                        violations.append(
                            f"gt/scan_index: indices must be in [0, {n_scans}); "
                            f"got [{scan_index.min()}, {scan_index.max()}]"
                        )
                    if len(set(scan_index.tolist())) != n_gt:
                        violations.append("gt/scan_index: indices must be unique")
                    if "splits/challenge_val" in f:
                        chal = set(f["splits/challenge_val"][...].tolist())
                        leaks = sorted(set(scan_index.tolist()) & chal)
                        if leaks:
                            violations.append(
                                f"gt/scan_index leaks {len(leaks)} challenge_val indices "
                                f"(first few: {leaks[:5]})"
                            )

            if "n_with_gt" in f.attrs and int(f.attrs["n_with_gt"]) != n_gt:
                violations.append(
                    f"root attr 'n_with_gt' = {int(f.attrs['n_with_gt'])} disagrees with "
                    f"latents/gt_anchor.shape[0] = {n_gt}"
                )

        # 4. CSR invariants
        violations.extend(_check_csr_invariants(f, n_scans))
        violations.extend(_check_augmentation_id_membership(f))

        # 5. Splits partition
        violations.extend(_check_split_partition(f, n_scans))

        # 6. Latent stats placeholders
        violations.extend(_check_latent_stats_shapes(f))

    return violations


def assert_brainrepa_latents_valid(path: str | Path) -> None:
    """Raise :class:`LatentsH5SchemaError` if the file does not conform to Schema B.

    Parameters:
        path: Filesystem path to the H5 to validate.

    Raises:
        LatentsH5SchemaError: With a joined violation list as the message.
        FileNotFoundError: If ``path`` does not exist.
    """
    violations = validate_brainrepa_latents(path)
    if violations:
        msg = f"{Path(path)} does not conform to LATENTS_SCHEMA:\n  - " + "\n  - ".join(violations)
        raise LatentsH5SchemaError(msg)


__all__ = ["assert_brainrepa_latents_valid", "validate_brainrepa_latents"]
