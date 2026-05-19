# H5 Schemas

This document is a *design note* for the two HDF5 schemas the BrainREPA-FM pipeline writes. The machine-readable specs live in
[`src/brainrepa_fm/data/h5_schemas.py`](../src/brainrepa_fm/data/h5_schemas.py). Both schemas satisfy every principle in
[`.claude/rules/h5-design-principles.md`](../.claude/rules/h5-design-principles.md): attrs-driven, validator-paired, write-then-assert, gzip-4, self-describing, splits-as-indices, CSR layout for variable-length groupings.

They are intentionally not MenFlow-compatible — BraTS-2026 is single-cohort, single-modality, cross-sectional.

## Schema A — `brats_inpainting_2026.h5`

**Producer:** `routines.data.brats2026_convert` (via [`src/brainrepa_fm/data/brats2026_converter.py`](../src/brainrepa_fm/data/brats2026_converter.py)).
**Validator:** [`src/brainrepa_fm/data/brats2026_schema.py::assert_brats2026_valid`](../src/brainrepa_fm/data/brats2026_schema.py).
**Output path (default):** `/media/mpascual/MeningD2/INPAINTING/2026/h5/brats_inpainting_2026.h5`.

### Root attributes

| Attr | Type | Description |
|---|---|---|
| `schema_version` | str | Currently `"1.0"`. Bumped on breaking changes. |
| `created_at` | str | ISO-8601 UTC string. |
| `producer` | str | e.g. `"routines.data.brats2026_convert:v0.0.1"`. |
| `config_json` | str | JSON-encoded converter config. |
| `git_sha` | str | Git commit SHA at producer-run time, or `"unknown"`. |
| `orientation` | str | Fixed `"RAS"`. |
| `voxel_spacing_mm` | str | JSON-encoded `[1.0, 1.0, 1.0]`. |
| `preprocessing` | str | One-line provenance string. |
| `n_scans` | int | Total leading-dim count. |

### Datasets

Always-present per-scan (leading dim = `n_scans`):

| Path | Dtype | Trailing shape | Source |
|---|---|---|---|
| `scan_id` | vlen-str | — | `BraTS-GLI-NNNNN-XXX`. |
| `cohort` | vlen-str | — | `"GLI"` for all rows currently. |
| `split` | vlen-str | — | One of `{train, val, test, challenge_val}`. |
| `metadata/source_path` | vlen-str | — | NIfTI subject directory. |
| `metadata/voxel_intensity_clip` | float32 | `(2,)` | Per-scan (p5, p99.5) used for normalization of both voided and GT T1. |
| `images/t1_voided` | float32 | `(240, 240, 155)` | `<sid>-t1n-voided.nii.gz`, 5th–99.5th percentile clip → min-max `[0, 1]`. The void region is exactly 0. |
| `masks/brain` | int8 | `(240, 240, 155)` | Computed: `(t1_voided > 0) | (void == 1)`. |
| `masks/void` | int8 | `(240, 240, 155)` | `<sid>-mask.nii.gz` (= `mask-healthy ∪ mask-unhealthy`). |

Sparse ground-truth group (leading dim = `n_with_gt`, training pool only):

| Path | Dtype | Trailing shape | Source |
|---|---|---|---|
| `gt/scan_index` | int32 | — | Pointers into the global `scan_id` axis (disjoint from `splits/challenge_val`). |
| `gt/t1` | float32 | `(240, 240, 155)` | `<sid>-t1n.nii.gz`, same per-scan clip as `images/t1_voided`. |
| `gt/healthy_mask` | int8 | `(240, 240, 155)` | `<sid>-mask-healthy.nii.gz`. Leaderboard scoring region. |
| `gt/tumor_mask` | int8 | `(240, 240, 155)` | `<sid>-mask-unhealthy.nii.gz`. Subset of `masks/void`. |

Splits (H5 principle §9):

| Path | Dtype | Notes |
|---|---|---|
| `splits/train` | int32 | Indices into `scan_id`. |
| `splits/val` | int32 | Local validation. |
| `splits/test` | int32 | Local held-out test. |
| `splits/challenge_val` | int32 | The 219 official BraTS-2026 challenge-val subjects (no tumor GT). |

### Cross-field invariants enforced by `validate_brats2026`

1. All required root attrs present and well-typed; `schema_version == "1.0"`; `config_json` parses as JSON.
2. Required datasets exist; per-scan leading dims agree; trailing shapes match `BRATS_VOLUME_SHAPE = (240, 240, 155)`.
3. `images/t1_voided` first-scan values in `[0, 1]`; `masks/{brain,void}` values ⊆ `{0, 1}`.
4. `cohort ⊆ ALLOWED_COHORTS = ("GLI",)`; `split ⊆ ALLOWED_SPLITS = ("train", "val", "test", "challenge_val")`.
5. `splits/*` indices: int32, unique per split, in `[0, n_scans)`, partition `range(n_scans)` exhaustively without overlap.
6. If any `gt/*` dataset is present, all four must be present with matching leading dim. `gt/scan_index` indices are unique, in range, and disjoint from `splits/challenge_val`. `gt/t1` first row in `[0, 1]`.

### Storage policy

- Compression: `gzip` level 4 on `images/t1`, `masks/{brain,void}`, `masks/tumor/values`.
- Chunking: `(1, ...trailing)` so one read = one scan.
- Dtypes: `float32` for intensities; `int8` for binary masks; `int32` for indices; vlen-str for IDs and paths.
- Atomic write: producer writes `out_path.partial`, calls `assert_brats2026_valid`, then `os.replace` into `out_path`.

## Schema B — `brainrepa_latents.h5` (forward-declared)

**Producer:** *future task* — runs after pre-flight 01's `decision.json` is finalized so that `augmentations/include` is known.
**Validator:** [`src/brainrepa_fm/data/brainrepa_latents_schema.py::assert_brainrepa_latents_valid`](../src/brainrepa_fm/data/brainrepa_latents_schema.py).

### Root attributes

In addition to Schema A's standard attrs (`schema_version`, `created_at`, `producer`, `config_json`, `git_sha`, `n_scans`):

| Attr | Type | Description |
|---|---|---|
| `n_with_gt` | int | Number of scans carrying a ground-truth T1 (== leading dim of `latents/gt_anchor`). |
| `latent_stats_calibrated` | bool | False at first write; True once pre-flight 03 populates `latent_scale` / `latent_mean`. |
| `vae_checkpoint_sha256` | str | First 16 chars of the SHA-256 of `autoencoder_v2.pt`. |
| `vae_scale_factor` | float | Scalar multiplier applied on encode (and divider on decode). |
| `paired_source` | str | Absolute path to the paired Schema A H5; their `scan_id` orderings must agree. |
| `latent_channels` | int | 4 for MAISI v2. |
| `latent_spatial_shape` | str | JSON `[Lz, Ly, Lx]` — pinned at H5 creation from the empirical MAISI probe. |

### Datasets

| Path | Dtype | Trailing shape | Notes |
|---|---|---|---|
| `scan_id` | vlen-str | — | Mirror of Schema A `scan_id` ordering. |
| `split` | vlen-str | — | Mirror. |
| `latents/voided_anchor` | float32 | `(C, Lz, Ly, Lx)` | Encoded `images/t1_voided` (= FM model input z̃). Always present, leading dim = `n_scans`. |
| `latents/gt_anchor` | float32 | `(C, Lz, Ly, Lx)` | Encoded `gt/t1` (= FM training target z₀). Sparse, leading dim = `n_with_gt`. |
| `gt/scan_index` | int32 | — | Pointers into `scan_id` for the rows in `latents/gt_anchor`. Disjoint from `splits/challenge_val`. |
| `latents/augmented/values` | float32 | `(C, Lz, Ly, Lx)` | CSR-stacked per-augmentation voided latents. Leading dim = `n_aug_rows`. |
| `latents/augmented/offsets` | int32 | — | Shape `(n_scans + 1,)`. Row range for scan i is `[offsets[i], offsets[i+1])`. |
| `latents/augmented/augmentation_ids` | vlen-str | — | Transform IDs in CSR order. |
| `augmentations/include` | vlen-str | — | Mirror of pre-flight 01's `decision.json::include`. |
| `latent_scale` | float32 | `(4,)` | Per-channel std for FM standardization. Zeros placeholder until calibrated. |
| `latent_mean` | float32 | `(4,)` | Per-channel mean. Zeros placeholder until calibrated. |
| `splits/{train,val,test,challenge_val}` | int32 | — | Same partition as Schema A. |

The two anchors mirror the SD-Inpainting / flow-matching formulation in the proposal: at training time the FM generator sees the conditioning `z̃ = latents/voided_anchor[i]` and learns to recover `z₀ = latents/gt_anchor[gt_to_global_inverse[i]]` along a stochastic interpolation. At inference time only `z̃` is needed, so `challenge_val` rows carry only `voided_anchor`.

### Cross-field invariants enforced by `validate_brainrepa_latents`

1. Required root attrs + valid JSON for `config_json` and `latent_spatial_shape`.
2. `latents/anchor.shape == (n_scans, latent_channels, Lz, Ly, Lx)` with `(Lz, Ly, Lx)` matching the `latent_spatial_shape` attr.
3. CSR invariants: `offsets.shape == (n_scans + 1,)`, monotonic non-decreasing, starts at 0; `len(values) == len(augmentation_ids) == offsets[-1]`.
4. Every `augmentation_ids` value appears in `augmentations/include`.
5. Splits partition `range(n_scans)` (same rule as Schema A).
6. `latent_scale.shape == latent_mean.shape == (4,)` with dtype `float32`.

### Note on the empirical latent spatial shape

The prompt declares `(Lz, Ly, Lx) ≈ (64, 64, 48)`. The MAISI config in
[`MAISI_V2_RM/code/NV-Generate-CTMR/configs/config_network_ddpm.json`](file:///media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR/configs/config_network_ddpm.json)
specifies `num_channels=[64, 128, 256]`. The number of downsample stages depends on the upstream `AutoencoderKlMaisi`
construction details, so the latent producer pins `latent_spatial_shape` from a single deterministic forward pass on a
`(1, 1, 256, 256, 192)` probe. If the value disagrees with the prompt, a `DECISIONS.md` entry records the correction.

## Why two H5s and not one?

The source H5 is *cohort × scan*-level and stable; new modeling work re-reads it. The latent H5 is *augmentation × scan*-level and version-coupled to the VAE checkpoint. Separating them keeps re-encoding cheap (one re-write touches only the latent file) and lets multiple latent producers (e.g. one frozen-VAE, one fine-tuned-VAE) coexist, paired to the same source.
