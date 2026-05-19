# HDF5 Design Principles

When this project produces an `.h5` artifact (intermediate cache, preflight output, latent dump, evaluation result), the file must satisfy the principles below. The principles are inspired by — but not identical to — the MenFlow unified H5 schema. We do not enforce that schema; each artifact owns its own layout and documents it in the module that creates it.

A new H5 producer in this codebase **must**:

1. **Write a `schema_version` root attribute** (string, e.g. `"1.0"`). Bump on any breaking change. Consumers branch on this.
2. **Write `created_at`** (ISO-8601 UTC string) and **`producer`** (string, e.g. `"routines.preflights.maisi_vae:v0.1.0"`) root attributes. Knowing *who* wrote a file and *when* is non-negotiable for reproducibility.
3. **Persist the producing config** as a JSON-encoded string attribute (`config_json`). The exact YAML used to invoke the routine round-trips into the artifact. Persist the git commit SHA in a separate `git_sha` attribute when available.
4. **Self-describe every dataset.** Each `/path/to/dataset` carries:
   - `units` — string, e.g. `"voxels"`, `"cm^3"`, `"dB"`, `"dimensionless"`
   - `description` — one sentence semantic meaning
   - `dtype` — echo of the actual dtype
   - `leading_dim` — string, e.g. `"n_scans"`, if applicable
   An analyst opening the file in `h5py` must not need to read source code to understand a field.
5. **Be cohort-agnostic at the schema level.** Cohort-specific metadata goes under a `metadata/` group as one dataset per field (`metadata/age`, `metadata/sex`, `metadata/grade`, …). The H5 layout itself does not hard-code field names from any one cohort.
6. **Storage policy:**
   - Compression: `gzip` level 4 on bulky datasets (`images`, `latents`, `segmentations`, `masks`).
   - Chunking: `(1, ...rest)` so reading one scan is one read.
   - Dtypes: `float32` for intensities/latents; `int8` for segmentation labels (range `[0, 127]`); `int32` for indices; vlen-str for IDs.
   - Native shape preferred. Do not resample or normalize at write time unless the conversion is irreversible by design (state this in the producer module).
7. **Cross-field invariants are enforced in a single validator** alongside the producer. Pattern:
   ```python
   def validate_<artifact>(path: Path) -> list[str]: ...   # returns violations
   def assert_<artifact>_valid(path: Path) -> None: ...    # raises if non-empty
   ```
   The producer calls `assert_<artifact>_valid(path)` **before** returning the path from `Engine.run()`. A non-conformant artifact must never reach disk in a "successful" state.
8. **CSR-style layout for variable-length groupings.** When grouping rows by a key (patient → scans, subject → sessions), store:
   - `<group>/offsets` — int32, length `n_groups + 1`, monotonic non-decreasing, starts at 0, ends at `n_rows`.
   - `<group>/keys` — vlen str, length `n_groups`, unique IDs in offset order.
   Avoid Python-side groupby at read time.
9. **Splits are indices, not boolean masks.** Patient-level or scan-level splits stored as `splits/<name>` (int32 indices into the corresponding ID list).
10. **Validate on open in consumers.** Every consumer asserts `schema_version` is in its supported set; mismatched files raise immediately with a clear message naming the producer.

## What this means for `brats_inpainting_2026.h5`

The BraTS 2026 H5 cache at `/media/mpascual/MeningD2/INPAINTING/2026/h5/brats_inpainting_2026.h5` is single-cohort, single-modality (T1), cross-sectional. Its schema is owned by the converter that produces it and lives in `src/brainrepa_fm/data/<converter>.py`. The schema is documented in the converter module's docstring and validated by an `assert_brats2026_valid(path)` helper, following principles 1-10 above. **It is not MenFlow-compatible** — it does not carry longitudinal groups or multi-modality channels — and that is by design.

When in doubt, prefer the structure of `menflow.data.h5_schema` in the MenFlow project as a reference implementation. The contract here is the *spirit* of that schema (attrs-driven, validator-paired, write-then-assert, cohort-agnostic), not its specific field names.
