"""Declarative HDF5 schema definitions.

Two schemas are defined here:

- ``BRATS2026_SCHEMA`` — the source H5 holding the 1,470 BraTS-GLI scans
  (1,251 training + 219 challenge-validation), per-scan T1 volume, brain mask,
  void mask, and (where available) tumor mask.
- ``LATENTS_SCHEMA`` — the latent H5 (forward-declared; producer is a follow-up
  task) holding per-scan MAISI-V2 VAE-GAN-encoded latents and per-augmentation
  latents for the transforms that passed pre-flight 01.

Both schemas satisfy every principle in ``.claude/rules/h5-design-principles.md``:
schema_version + created_at + producer + config_json + git_sha root attrs,
self-describing datasets (units, description, dtype, leading_dim attrs),
CSR-style layout for variable-length groupings, splits-as-indices, validator
paired with the producer.

This module is import-cheap (no h5py, no numpy ops at import time). The
validators in :mod:`brainrepa_fm.data.brats2026_schema` and
:mod:`brainrepa_fm.data.brainrepa_latents_schema` consume the dataclasses
defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Native BraTS-2023/2026 T1 volume shape (RAS, 1 mm isotropic).
BRATS_VOLUME_SHAPE: tuple[int, int, int] = (240, 240, 155)

# Cohort tag used in the ``cohort`` dataset. Currently single-cohort (GLI) but
# the field is kept for the Wk-8 cross-pathology stage (proposal §4).
ALLOWED_COHORTS: tuple[str, ...] = ("GLI",)

# Allowed values of the ``split`` dataset. ``challenge_val`` are the 219 official
# validation subjects (no tumor GT). ``train`` / ``val`` / ``test`` are a
# patient-level partition of the 1,251 training pool — see
# ``src/brainrepa_fm/data/brats_partition.py``.
ALLOWED_SPLITS: tuple[str, ...] = ("train", "val", "test", "challenge_val")

# Schema-version string written as a root attr. Bump on any breaking change.
BRATS2026_SCHEMA_VERSION: str = "1.0"
LATENTS_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Dataclasses describing the schema declaratively.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Single HDF5 dataset.

    Attributes:
        path: Slash-delimited dataset path (e.g. ``"images/t1"``).
        dtype: Expected NumPy dtype string (``"float32"``, ``"int8"``, ``"int32"``,
            or ``"vlen-str"`` for variable-length strings).
        trailing_shape: Expected per-row shape (everything after the leading
            ``n_scans`` axis). ``None`` for 1-D datasets where the leading axis
            *is* the only axis.
        units: Self-description tag (principle §4).
        description: One-sentence semantic meaning.
        leading_dim: Name of the leading axis (``"n_scans"`` for per-scan
            datasets, etc.). ``""`` for root-scalar-vector datasets.
        required: If False, the dataset may be absent (e.g. ``masks/tumor``
            for ``challenge_val`` scans).
    """

    path: str
    dtype: str
    trailing_shape: tuple[int, ...] | None
    units: str
    description: str
    leading_dim: str
    required: bool = True


@dataclass(frozen=True)
class AttrSpec:
    """Single HDF5 attribute (root- or group-level).

    Attributes:
        name: Attribute key.
        dtype: ``"str"``, ``"bool"``, ``"float"``, or ``"int"``.
        description: Why this attr exists (for `report.md` consumption).
        required: If False, may be absent.
    """

    name: str
    dtype: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class H5SchemaSpec:
    """Full HDF5 schema for one producer.

    Attributes:
        name: Schema identifier (used in error messages).
        schema_version: Version string written to the root attr.
        root_attrs: Root-level attribute specs.
        datasets: Per-scan and root-level dataset specs.
        allowed_splits: Allowed values of the ``split`` dataset (cross-checked
            by the validator). Empty tuple if the schema has no ``split``.
    """

    name: str
    schema_version: str
    root_attrs: tuple[AttrSpec, ...]
    datasets: tuple[DatasetSpec, ...]
    allowed_splits: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Schema A — ``brats_inpainting_2026.h5``
# ---------------------------------------------------------------------------


BRATS2026_ROOT_ATTRS: tuple[AttrSpec, ...] = (
    AttrSpec("schema_version", "str", "Producer schema version. Bumped on breaking changes."),
    AttrSpec("created_at", "str", "ISO-8601 UTC timestamp of producer completion."),
    AttrSpec(
        "producer", "str", "Producer identifier, e.g. 'routines.data.brats2026_convert:v0.0.1'."
    ),
    AttrSpec("config_json", "str", "JSON-encoded converter config that produced the file."),
    AttrSpec("git_sha", "str", "Git commit SHA at producer-run time, or 'unknown'."),
    AttrSpec("orientation", "str", "Volume orientation tag — fixed at 'RAS'."),
    AttrSpec("voxel_spacing_mm", "str", "JSON-encoded voxel spacing in millimetres."),
    AttrSpec(
        "preprocessing", "str", "One-line provenance string describing intensity normalization."
    ),
    AttrSpec("n_scans", "int", "Total number of scans in the file (== leading dimension)."),
)

BRATS2026_DATASETS: tuple[DatasetSpec, ...] = (
    # Per-scan ID / metadata fields (leading dim = n_scans).
    DatasetSpec(
        "scan_id",
        "vlen-str",
        None,
        "id",
        "Per-scan unique identifier (BraTS-GLI-NNNNN-XXX).",
        "n_scans",
    ),
    DatasetSpec(
        "cohort", "vlen-str", None, "tag", "Cohort tag; currently 'GLI' for all rows.", "n_scans"
    ),
    DatasetSpec(
        "split",
        "vlen-str",
        None,
        "tag",
        "Split assignment in {train, val, test, challenge_val}.",
        "n_scans",
    ),
    DatasetSpec(
        "metadata/source_path",
        "vlen-str",
        None,
        "path",
        "Filesystem path to the original NIfTI directory.",
        "n_scans",
    ),
    DatasetSpec(
        "metadata/voxel_intensity_clip",
        "float32",
        (2,),
        "intensity",
        "Per-scan (5th, 99.5th) percentile values used for brain-mask normalization.",
        "n_scans",
    ),
    # Always-present per-scan: voided T1 (model input at inference) + brain + void.
    DatasetSpec(
        "images/t1_voided",
        "float32",
        BRATS_VOLUME_SHAPE,
        "dimensionless",
        "Voided T1 (model input). 5th-99.5th percentile clip inside (brain & ~void) then min-max to [0, 1]; "
        "the void region is exactly zero by construction.",
        "n_scans",
    ),
    DatasetSpec(
        "masks/brain",
        "int8",
        BRATS_VOLUME_SHAPE,
        "binary",
        "Brain support mask, computed as ((t1_voided > 0) | (void == 1)). 1 = brain, 0 = background.",
        "n_scans",
    ),
    DatasetSpec(
        "masks/void",
        "int8",
        BRATS_VOLUME_SHAPE,
        "binary",
        "Inpainting void mask m_v (1 = region to inpaint, 0 = observed). Equals mask-healthy ∪ mask-unhealthy.",
        "n_scans",
    ),
    # Sparse ground-truth group (training subjects only: full T1 + leaderboard scoring mask + tumor mask).
    DatasetSpec(
        "gt/scan_index",
        "int32",
        None,
        "index",
        "Indices into scan_id pointing at rows that carry ground truth (training pool only; "
        "disjoint from splits/challenge_val).",
        "n_with_gt",
    ),
    DatasetSpec(
        "gt/t1",
        "float32",
        BRATS_VOLUME_SHAPE,
        "dimensionless",
        "Ground-truth T1 (no void), normalized with the same per-scan clip as images/t1_voided.",
        "n_with_gt",
    ),
    DatasetSpec(
        "gt/healthy_mask",
        "int8",
        BRATS_VOLUME_SHAPE,
        "binary",
        "Synthetic healthy void (mask-healthy.nii.gz). The BraTS leaderboard scores inpainting on this region.",
        "n_with_gt",
    ),
    DatasetSpec(
        "gt/tumor_mask",
        "int8",
        BRATS_VOLUME_SHAPE,
        "binary",
        "Tumor (unhealthy) void mask (mask-unhealthy.nii.gz). Subset of masks/void.",
        "n_with_gt",
    ),
    # Patient-level splits as int32 indices (H5 principle §9).
    DatasetSpec(
        "splits/train",
        "int32",
        None,
        "index",
        "Indices into scan_id of the train partition.",
        "n_train",
    ),
    DatasetSpec(
        "splits/val",
        "int32",
        None,
        "index",
        "Indices into scan_id of the local val partition.",
        "n_val",
    ),
    DatasetSpec(
        "splits/test",
        "int32",
        None,
        "index",
        "Indices into scan_id of the local test partition.",
        "n_test",
    ),
    DatasetSpec(
        "splits/challenge_val",
        "int32",
        None,
        "index",
        "Indices into scan_id of the official BraTS-2026 challenge validation set (no tumor GT).",
        "n_challenge_val",
    ),
)

BRATS2026_SCHEMA: H5SchemaSpec = H5SchemaSpec(
    name="brats_inpainting_2026.h5",
    schema_version=BRATS2026_SCHEMA_VERSION,
    root_attrs=BRATS2026_ROOT_ATTRS,
    datasets=BRATS2026_DATASETS,
    allowed_splits=ALLOWED_SPLITS,
)


# ---------------------------------------------------------------------------
# Schema B — ``brainrepa_latents.h5`` (forward-declared)
# ---------------------------------------------------------------------------


LATENTS_ROOT_ATTRS: tuple[AttrSpec, ...] = (
    AttrSpec("schema_version", "str", "Producer schema version."),
    AttrSpec("created_at", "str", "ISO-8601 UTC timestamp."),
    AttrSpec("producer", "str", "Producer identifier."),
    AttrSpec("config_json", "str", "JSON-encoded producer config."),
    AttrSpec("git_sha", "str", "Git commit SHA."),
    AttrSpec("n_scans", "int", "Total number of scans (== leading dimension of latents/anchor)."),
    AttrSpec(
        "latent_stats_calibrated",
        "bool",
        "True iff latent_scale/latent_mean carry real values produced by pre-flight 03; "
        "False at first write (zeros placeholder).",
    ),
    AttrSpec(
        "vae_checkpoint_sha256", "str", "SHA-256 (first 16 chars) of the MAISI VAE checkpoint used."
    ),
    AttrSpec(
        "vae_scale_factor",
        "float",
        "MAISI scalar 'scale_factor' applied on encode (multiplied) and decode (divided).",
    ),
    AttrSpec(
        "paired_source",
        "str",
        "Absolute path to the paired source H5 (Schema A) whose scan_id ordering matches this file.",
        required=False,
    ),
    AttrSpec("latent_channels", "int", "Number of latent channels (4 for MAISI v2)."),
    AttrSpec("latent_spatial_shape", "str", "JSON-encoded (Lz, Ly, Lx) latent spatial shape."),
)

LATENTS_DATASETS: tuple[DatasetSpec, ...] = (
    # Per-scan ID/metadata mirror of Schema A.
    DatasetSpec(
        "scan_id",
        "vlen-str",
        None,
        "id",
        "Per-scan unique identifier, mirroring Schema A.",
        "n_scans",
    ),
    DatasetSpec(
        "split", "vlen-str", None, "tag", "Split assignment, mirroring Schema A.", "n_scans"
    ),
    # Anchor latent (no augmentation): leading dim = n_scans, trailing shape pinned at H5 creation.
    DatasetSpec(
        "latents/anchor",
        "float32",
        None,  # trailing shape pinned dynamically (e.g. (4, 64, 64, 48) or (4, 32, 32, 24))
        "dimensionless",
        "Anchor latent z_mu * scale_factor for the no-augmentation pass. Trailing shape (C, Lz, Ly, Lx) "
        "is set at producer-creation time from the empirical MAISI probe.",
        "n_scans",
    ),
    # CSR-style augmented latents (H5 principle §8): variable views per scan.
    DatasetSpec(
        "latents/augmented/values",
        "float32",
        None,
        "dimensionless",
        "Concatenation of per-augmentation latents across all scans (CSR-style). Trailing shape matches latents/anchor.",
        "n_aug_rows",
    ),
    DatasetSpec(
        "latents/augmented/offsets",
        "int32",
        None,
        "index",
        "CSR offsets into latents/augmented/values: row range for scan i is [offsets[i], offsets[i+1]).",
        "n_scans_plus_one",
    ),
    DatasetSpec(
        "latents/augmented/augmentation_ids",
        "vlen-str",
        None,
        "tag",
        "Transform ID (e.g. 'A.1', 'C.2') for each row in latents/augmented/values, in CSR order.",
        "n_aug_rows",
    ),
    # Augmentation roster.
    DatasetSpec(
        "augmentations/include",
        "vlen-str",
        None,
        "tag",
        "Mirror of the `include` list in pre-flight 01's decision.json at producer-run time.",
        "n_augmentations",
    ),
    # Root-level latent statistics (filled by pre-flight 03; zeros placeholder until then).
    DatasetSpec(
        "latent_scale",
        "float32",
        (4,),
        "dimensionless",
        "Per-channel latent std used to standardize prior to FM training. Zeros placeholder until calibrated.",
        "",
    ),
    DatasetSpec(
        "latent_mean",
        "float32",
        (4,),
        "dimensionless",
        "Per-channel latent mean used to standardize prior to FM training. Zeros placeholder until calibrated.",
        "",
    ),
    # Splits mirror (consumers cross-reference scan IDs).
    DatasetSpec("splits/train", "int32", None, "index", "Train indices into scan_id.", "n_train"),
    DatasetSpec("splits/val", "int32", None, "index", "Val indices into scan_id.", "n_val"),
    DatasetSpec("splits/test", "int32", None, "index", "Test indices into scan_id.", "n_test"),
    DatasetSpec(
        "splits/challenge_val",
        "int32",
        None,
        "index",
        "Challenge-val indices into scan_id.",
        "n_challenge_val",
    ),
)

LATENTS_SCHEMA: H5SchemaSpec = H5SchemaSpec(
    name="brainrepa_latents.h5",
    schema_version=LATENTS_SCHEMA_VERSION,
    root_attrs=LATENTS_ROOT_ATTRS,
    datasets=LATENTS_DATASETS,
    allowed_splits=ALLOWED_SPLITS,
)


__all__ = [
    "ALLOWED_COHORTS",
    "ALLOWED_SPLITS",
    "BRATS2026_SCHEMA",
    "BRATS2026_SCHEMA_VERSION",
    "BRATS_VOLUME_SHAPE",
    "LATENTS_SCHEMA",
    "LATENTS_SCHEMA_VERSION",
    "AttrSpec",
    "DatasetSpec",
    "H5SchemaSpec",
]
