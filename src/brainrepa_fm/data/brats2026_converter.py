"""NIfTI → ``brats_inpainting_2026.h5`` converter.

The producer for Schema A in :mod:`brainrepa_fm.data.h5_schemas`. Reads the
BraTS-GLI Local-Synthesis Challenge subjects from disk, computes a brain mask,
normalizes both the voided and (when present) ground-truth T1 inside the brain
mask via a 5th-99.5th percentile clip followed by min-max scaling to ``[0, 1]``,
persists the result with the storage policy required by
``.claude/rules/h5-design-principles.md``, and validates the file before
returning the path.

Per-subject file layout assumed (from inspection of the on-disk dataset, 2026-05-19):

Training subjects (``ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training``):
    BraTS-GLI-NNNNN-XXX/
        BraTS-GLI-NNNNN-XXX-t1n.nii.gz             # ground-truth T1
        BraTS-GLI-NNNNN-XXX-t1n-voided.nii.gz      # T1 with the void zeroed
        BraTS-GLI-NNNNN-XXX-mask.nii.gz            # full void (= healthy ∪ unhealthy)
        BraTS-GLI-NNNNN-XXX-mask-healthy.nii.gz    # leaderboard scoring region
        BraTS-GLI-NNNNN-XXX-mask-unhealthy.nii.gz  # tumor region

Challenge-validation subjects (``...Challenge-Validation``):
    BraTS-GLI-NNNNN-XXX/
        BraTS-GLI-NNNNN-XXX-t1n-voided.nii.gz
        BraTS-GLI-NNNNN-XXX-mask.nii.gz
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import h5py
import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field, field_validator

from brainrepa_fm.data.brats2026_schema import assert_brats2026_valid
from brainrepa_fm.data.brats_partition import partition_patients
from brainrepa_fm.data.exceptions import ConverterError
from brainrepa_fm.data.h5_schemas import (
    BRATS2026_SCHEMA_VERSION,
    BRATS_VOLUME_SHAPE,
)

logger = logging.getLogger(__name__)

# Producer identifier persisted in the root ``producer`` attribute.
PRODUCER_ID: str = "routines.data.brats2026_convert:v0.0.1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class BraTS2026ConvertConfig(BaseModel):
    """Frozen configuration for the BraTS-2026 → H5 converter.

    Attributes:
        training_root: Filesystem path to the BraTS training subjects directory.
        challenge_val_root: Filesystem path to the challenge-validation subjects.
        output_path: Destination ``.h5`` path. The converter writes to a sibling
            ``.partial`` file and renames atomically on validate-success.
        max_subjects_training: Cap the number of training subjects ingested.
            ``None`` for the full set. Used by smoke configs.
        max_subjects_challenge_val: Same, for challenge-val.
        partition_fractions: ``(train, val, test)`` patient-level fractions over
            the *training* pool. ``challenge_val`` is handled separately.
        partition_seed: Seed for the patient-level shuffle.
        percentile_low: Lower percentile for intensity clipping (default 5.0).
        percentile_high: Upper percentile (default 99.5).
        gzip_level: HDF5 gzip level for bulk datasets.
        log_level: Logging level for the converter run (`INFO`/`DEBUG`).
    """

    training_root: Path
    challenge_val_root: Path
    output_path: Path
    max_subjects_training: int | None = None
    max_subjects_challenge_val: int | None = None
    partition_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    partition_seed: int = 2026
    percentile_low: float = Field(default=5.0, ge=0.0, le=100.0)
    percentile_high: float = Field(default=99.5, ge=0.0, le=100.0)
    gzip_level: int = Field(default=4, ge=0, le=9)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("training_root", "challenge_val_root")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        p = Path(v).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"path does not exist: {p}")
        return p

    @field_validator("output_path")
    @classmethod
    def _resolve_out(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()


# ---------------------------------------------------------------------------
# Per-subject record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRecord:
    """One BraTS subject discovered on disk."""

    scan_id: str
    source_dir: Path
    has_gt: bool


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class BraTS2026Converter:
    """Producer for ``brats_inpainting_2026.h5`` (Schema A).

    Parameters:
        config: Frozen converter configuration.
    """

    def __init__(self, config: BraTS2026ConvertConfig) -> None:
        self.config = config
        logging.basicConfig(level=config.log_level)
        logger.setLevel(config.log_level)

    # -- public ------------------------------------------------------------

    def run(self) -> Path:
        """Produce the H5 file at ``config.output_path`` and return that path.

        The file is first written to ``{output_path}.partial`` and renamed on
        validate-success. A non-conformant artifact never reaches the final path.

        Returns:
            The absolute path to the validated H5.

        Raises:
            ConverterError: On I/O / source-data problems.
            BratsH5SchemaError: From the post-write validator.
        """
        training = self._discover_subjects(
            self.config.training_root, has_gt=True, cap=self.config.max_subjects_training
        )
        chal_val = self._discover_subjects(
            self.config.challenge_val_root,
            has_gt=False,
            cap=self.config.max_subjects_challenge_val,
        )
        records: list[SubjectRecord] = training + chal_val
        if not records:
            raise ConverterError("no subjects discovered in either training or challenge-val roots")

        train_scan_ids = [r.scan_id for r in training]
        partitions = partition_patients(
            train_scan_ids,
            fractions=self.config.partition_fractions,
            seed=self.config.partition_seed,
        )
        # partition_patients indexes into training only; offsets shift into the
        # global record list (challenge-val rows are appended after training).
        n_training = len(training)
        challenge_val_idx = np.arange(n_training, n_training + len(chal_val), dtype=np.int32)
        splits = {**partitions, "challenge_val": challenge_val_idx}

        logger.info(
            "discovered %d training + %d challenge-val subjects; splits = %s",
            n_training,
            len(chal_val),
            {k: int(v.size) for k, v in splits.items()},
        )

        out_path = self.config.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(out_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()

        try:
            self._write_h5(partial, records, splits)
            assert_brats2026_valid(partial)
        except Exception:
            if partial.exists():
                partial.unlink(missing_ok=True)
            raise

        os.replace(partial, out_path)
        logger.info("wrote and validated %s", out_path)
        return out_path

    # -- subject discovery ------------------------------------------------

    @staticmethod
    def _discover_subjects(root: Path, *, has_gt: bool, cap: int | None) -> list[SubjectRecord]:
        records: list[SubjectRecord] = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            scan_id = d.name
            # minimal presence check
            required = [
                f"{scan_id}-t1n-voided.nii.gz",
                f"{scan_id}-mask.nii.gz",
            ]
            if has_gt:
                required.extend(
                    [
                        f"{scan_id}-t1n.nii.gz",
                        f"{scan_id}-mask-healthy.nii.gz",
                        f"{scan_id}-mask-unhealthy.nii.gz",
                    ]
                )
            missing = [name for name in required if not (d / name).exists()]
            if missing:
                logger.warning("skipping %s: missing %s", scan_id, missing)
                continue
            records.append(SubjectRecord(scan_id=scan_id, source_dir=d, has_gt=has_gt))
            if cap is not None and len(records) >= cap:
                break
        return records

    # -- I/O --------------------------------------------------------------

    def _load_subject(
        self, rec: SubjectRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Load NIfTIs for one subject; arrays are NOT normalized yet.

        Returns:
            (t1_voided, void_mask, t1_gt, healthy_mask, tumor_mask) — the last three
            are ``None`` for challenge-val subjects.
        """
        sid = rec.scan_id
        d = rec.source_dir

        t1_voided = self._load_nifti_volume(d / f"{sid}-t1n-voided.nii.gz", expect_binary=False)
        void = self._load_nifti_volume(d / f"{sid}-mask.nii.gz", expect_binary=True).astype(np.int8)

        if rec.has_gt:
            t1_gt = self._load_nifti_volume(d / f"{sid}-t1n.nii.gz", expect_binary=False)
            healthy = self._load_nifti_volume(
                d / f"{sid}-mask-healthy.nii.gz", expect_binary=True
            ).astype(np.int8)
            tumor = self._load_nifti_volume(
                d / f"{sid}-mask-unhealthy.nii.gz", expect_binary=True
            ).astype(np.int8)
        else:
            t1_gt = healthy = tumor = None
        return t1_voided, void, t1_gt, healthy, tumor

    @staticmethod
    def _load_nifti_volume(path: Path, *, expect_binary: bool) -> np.ndarray:
        img = nib.load(path)
        arr = np.asarray(img.dataobj)
        if arr.shape != BRATS_VOLUME_SHAPE:
            raise ConverterError(
                f"{path}: shape {arr.shape} != BRATS_VOLUME_SHAPE {BRATS_VOLUME_SHAPE}"
            )
        if expect_binary:
            return (arr > 0.5).astype(np.int8)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _compute_brain_mask(t1_voided: np.ndarray, void: np.ndarray) -> np.ndarray:
        """Brain mask = (voided > 0) ∪ (void == 1)."""
        return ((t1_voided > 0) | (void.astype(bool))).astype(np.int8)

    def _normalize_t1(
        self,
        t1_voided: np.ndarray,
        brain: np.ndarray,
        void: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Brain-mask 5th-99.5th percentile clip → min-max [0, 1] on the voided T1.

        Returns:
            (normalized voided T1, (p_low, p_high) used).
        """
        observed = (brain.astype(bool)) & (~void.astype(bool))
        sample = t1_voided[observed]
        if sample.size == 0:
            raise ConverterError("normalization failed: empty observed mask (brain ∩ ~void)")
        p_low = float(np.percentile(sample, self.config.percentile_low))
        p_high = float(np.percentile(sample, self.config.percentile_high))
        if p_high <= p_low:
            raise ConverterError(
                f"normalization failed: degenerate percentiles p{self.config.percentile_low}="
                f"{p_low:.4f}, p{self.config.percentile_high}={p_high:.4f}"
            )
        norm = (np.clip(t1_voided, p_low, p_high) - p_low) / (p_high - p_low)
        norm = norm.astype(np.float32, copy=False)
        # Force exact zero inside the void (clipping a 0 sample may produce a positive value).
        norm[void.astype(bool)] = 0.0
        # Force exact zero outside the brain.
        norm[~brain.astype(bool)] = 0.0
        return norm, (p_low, p_high)

    @staticmethod
    def _normalize_gt(
        t1_gt: np.ndarray, brain: np.ndarray, clip: tuple[float, float]
    ) -> np.ndarray:
        p_low, p_high = clip
        norm = (np.clip(t1_gt, p_low, p_high) - p_low) / (p_high - p_low)
        norm = norm.astype(np.float32, copy=False)
        norm[~brain.astype(bool)] = 0.0
        return norm

    # -- write ------------------------------------------------------------

    def _write_h5(
        self, path: Path, records: list[SubjectRecord], splits: dict[str, np.ndarray]
    ) -> None:
        n = len(records)
        gt_indices = [i for i, r in enumerate(records) if r.has_gt]
        n_with_gt = len(gt_indices)

        config_json = json.dumps(asdict_config(self.config), sort_keys=True)

        with h5py.File(path, "w") as f:
            # Root attrs
            f.attrs["schema_version"] = BRATS2026_SCHEMA_VERSION
            f.attrs["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            f.attrs["producer"] = PRODUCER_ID
            f.attrs["config_json"] = config_json
            f.attrs["git_sha"] = _git_sha()
            f.attrs["orientation"] = "RAS"
            f.attrs["voxel_spacing_mm"] = json.dumps([1.0, 1.0, 1.0])
            f.attrs["preprocessing"] = (
                "5th-99.5th percentile clip inside (brain & ~void) → min-max [0, 1]; "
                "void region forced to exactly 0."
            )
            f.attrs["n_scans"] = n

            # Per-scan ID datasets
            vlen_str = h5py.string_dtype(encoding="utf-8")
            scan_id_ds = f.create_dataset("scan_id", shape=(n,), dtype=vlen_str)
            cohort_ds = f.create_dataset("cohort", shape=(n,), dtype=vlen_str)
            split_ds = f.create_dataset("split", shape=(n,), dtype=vlen_str)
            source_ds = f.create_dataset("metadata/source_path", shape=(n,), dtype=vlen_str)
            clip_ds = f.create_dataset(
                "metadata/voxel_intensity_clip", shape=(n, 2), dtype=np.float32
            )

            # Volume + masks
            chunk = (1, *BRATS_VOLUME_SHAPE)
            t1_voided_ds = f.create_dataset(
                "images/t1_voided",
                shape=(n, *BRATS_VOLUME_SHAPE),
                dtype=np.float32,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )
            brain_ds = f.create_dataset(
                "masks/brain",
                shape=(n, *BRATS_VOLUME_SHAPE),
                dtype=np.int8,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )
            void_ds = f.create_dataset(
                "masks/void",
                shape=(n, *BRATS_VOLUME_SHAPE),
                dtype=np.int8,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )

            # GT group (sparse)
            gt_scan_index_ds = f.create_dataset("gt/scan_index", shape=(n_with_gt,), dtype=np.int32)
            gt_t1_ds = f.create_dataset(
                "gt/t1",
                shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
                dtype=np.float32,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )
            gt_healthy_ds = f.create_dataset(
                "gt/healthy_mask",
                shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
                dtype=np.int8,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )
            gt_tumor_ds = f.create_dataset(
                "gt/tumor_mask",
                shape=(n_with_gt, *BRATS_VOLUME_SHAPE),
                dtype=np.int8,
                chunks=chunk,
                compression="gzip",
                compression_opts=self.config.gzip_level,
            )

            # Split membership
            split_of = np.empty(n, dtype=object)
            for name, idx in splits.items():
                split_of[idx] = name
            if any(s is None for s in split_of):
                raise ConverterError("internal: some scans were not assigned a split")

            # Streaming write
            gt_cursor = 0
            for i, rec in enumerate(records):
                t1v, void, t1_gt, healthy, tumor = self._load_subject(rec)
                brain = self._compute_brain_mask(t1v, void)
                t1v_norm, clip = self._normalize_t1(t1v, brain, void)

                scan_id_ds[i] = rec.scan_id
                cohort_ds[i] = "GLI"
                split_ds[i] = str(split_of[i])
                source_ds[i] = str(rec.source_dir)
                clip_ds[i] = np.asarray(clip, dtype=np.float32)
                t1_voided_ds[i] = t1v_norm
                brain_ds[i] = brain
                void_ds[i] = void

                if rec.has_gt:
                    assert t1_gt is not None and healthy is not None and tumor is not None
                    gt_norm = self._normalize_gt(t1_gt, brain, clip)
                    gt_scan_index_ds[gt_cursor] = i
                    gt_t1_ds[gt_cursor] = gt_norm
                    gt_healthy_ds[gt_cursor] = healthy
                    gt_tumor_ds[gt_cursor] = tumor
                    gt_cursor += 1

                if (i + 1) % 25 == 0 or i + 1 == n:
                    logger.info("wrote scan %d/%d (%s)", i + 1, n, rec.scan_id)

            # Splits / partition
            for name in ("train", "val", "test", "challenge_val"):
                f.create_dataset(f"splits/{name}", data=np.asarray(splits[name], dtype=np.int32))

            # Self-description attrs on every dataset (principle §4).
            _add_dataset_attrs(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def asdict_config(cfg: BraTS2026ConvertConfig) -> dict[str, object]:
    """Pydantic config → JSON-safe dict (Path → str)."""
    d = cfg.model_dump()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _add_dataset_attrs(f: h5py.File) -> None:
    """Attach (units, description, dtype, leading_dim) attrs per principle §4.

    Pulled from the declarative spec in :mod:`brainrepa_fm.data.h5_schemas`.
    """
    from brainrepa_fm.data.h5_schemas import BRATS2026_SCHEMA

    for spec in BRATS2026_SCHEMA.datasets:
        if spec.path not in f:
            continue
        d = f[spec.path]
        d.attrs["units"] = spec.units
        d.attrs["description"] = spec.description
        d.attrs["dtype"] = spec.dtype
        d.attrs["leading_dim"] = spec.leading_dim


__all__ = ["PRODUCER_ID", "BraTS2026ConvertConfig", "BraTS2026Converter"]
