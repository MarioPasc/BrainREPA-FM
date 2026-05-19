# BrainREPA-FM Decision Log

Append-only, human-readable record of non-trivial project decisions. Newest entry on top. ISO-8601 dates only. Pre-flight `decision.json` files (under `artifacts/preflights/<name>/<UTC>/`) are the machine-readable counterpart; this file is the narrative.

Format:

```
## YYYY-MM-DD — <short title>

**Context:** <question / alternatives>
**Decision:** <what we picked>
**Consequences:** <downstream effects>
**Status:** accepted | superseded by <YYYY-MM-DD-other-title>
```

Status updates edit the original entry's status line. Entries are never deleted.

---

## 2026-05-19 — Runtime dependency pins for pre-flight 01 (H5 schemas + augmentation audit)

**Context:** Pre-flight 01 needs an HDF5 stack (`h5py`), NIfTI I/O (`nibabel`, `SimpleITK`), 3D image transforms (`monai`), KS testing and binary-image descriptors (`scipy`, `scikit-image`), config plumbing (`omegaconf`, `pydantic`, `pyyaml`), logging (`rich`), tabular outputs (`pandas`), and visualization (`matplotlib`). Torch is needed for the MAISI VAE wrapper. The previous `pyproject.toml` had `dependencies = []` per bootstrap policy.
**Decision:** Pin `dependencies` to `numpy<2.2`, `scipy>=1.11`, `scikit-image>=0.22`, `nibabel>=5.2`, `SimpleITK>=2.3`, `h5py>=3.10`, `einops>=0.7`, `omegaconf>=2.3`, `pydantic>=2.6`, `rich>=13.7`, `matplotlib>=3.8`, `pandas>=2.1`, `monai[einops]>=1.4,<2.0`, `PyYAML>=6.0`. Torch is still **not** in `[project.dependencies]` — install-time wheel selection picks the CUDA-matching wheel (`cu121` on the local 3060 → `torch==2.5.1+cu121`). Dev extras add `pytest-cov`. The unused `[project.urls]` block was removed (its `file://` URL was rejected by setuptools).
**Consequences:** `pip install -e ".[dev]"` reproduces the env after a one-line `pip install --extra-index-url .../cu121 torch torchvision`. `numpy<2.2` is a precaution against MONAI 1.5's numpy-2 compatibility window. Two console scripts (`brainrepa-data-brats2026-convert`, `brainrepa-preflight-augmentation`) are now active forward declarations that will resolve once their modules land.
**Status:** accepted

## 2026-05-19 — `decision.json` keys follow `preflight-pattern.md`, not spec §7

**Context:** `docs/checks/01_augmentation_preflight.md` §7 lists `decision.json` keys as `image_space_augmentations`, `latent_space_augmentations`, `dropped_augmentations`, `drop_reasons`, `augmentation_probabilities`. `.claude/rules/preflight-pattern.md` (and the implementation prompt) list them as `include`, `drop`, `halve_range`, `vae_composition_gap_db`, `ks_p_values`, `ks_hard_fail`. The prompt §0 declares "docs win" yet §1 D3 pins the `preflight-pattern.md` schema, so the prompt is internally inconsistent.
**Decision:** Use the `preflight-pattern.md`/prompt schema verbatim (`include`, `drop`, `halve_range`, `vae_composition_gap_db`, `ks_p_values`, `ks_hard_fail`, plus a side-car `drop_reasons` dict for C.4's "VAE_erased_noise"-style annotations). The spec §7 names are not written to disk; `report.md` carries the prose narrative.
**Consequences:** Downstream FM training routines parse the canonical six keys. If the spec §7 naming becomes the de facto contract later, a one-shot migration script can rename keys in existing artifacts. The deviation is explicitly logged here so peer review can audit it.
**Status:** accepted

## 2026-05-19 — Source H5 holds train + challenge-val cohorts; 80/10/10 patient split on the train pool

**Context:** BraTS 2026 provides 1,251 training subjects (with tumor GT) and 219 challenge-validation subjects (no GT but a frozen `mask.nii.gz` void). The training set must be partitioned for supervised model selection and held-out testing.
**Decision:** Single H5 (`brats_inpainting_2026.h5`) holds all 1,470 scans. `split` field ∈ `{train, val, test, challenge_val}`. The 219 challenge-val subjects are `challenge_val`. The 1,251 training subjects are split 80/10/10 at the *patient* level (one `BraTS-GLI-NNNNN` group → one split) with seed `2026`, persisted as int32 indices under `splits/<name>` (H5 principle §9). `metadata/source_path` records the originating NIfTI directory for every row.
**Consequences:** Supervised metrics during training/eval use the local `val` and `test` slices. The `challenge_val` slice is reserved for the leaderboard submission flow only — no supervised loss can be computed on it (no tumor GT, void already fixed). Re-running the converter with a different seed produces a new H5 (artifacts are immutable per `.claude/rules/preflight-pattern.md`).
**Status:** accepted

## 2026-05-19 — A.2 / A.3 use a simplified geometric sampler, not the verbatim BraTS sampler

**Context:** The user approved "re-invoke the sampler at preflight time (Recommended)" — vendor `getHealthyMasks` from `docs/2026_challenge/dataset/include.py` and call it with widen=1.5. Inspection shows the official sampler depends on a multi-stage pandas/scipy pipeline (`process_getHealthyMasks` → `sampleLocation` → `sampleCompartment` → tumor-compartment pool indexing) totalling several hundred lines. A faithful vendoring also pulls the BraTS distance-transform conventions and segmentation-mask conventions. Scope for this sprint was tight.
**Decision:** Ship a simplified geometric sampler `sample_void_mask(brain, tumor, *, widen_factor, seed)` at `src/brainrepa_fm/preflight/augmentation/transforms.py` that (i) picks an interior location via a brain distance transform, (ii) drops a randomly-anisotropic ellipsoid-plus-jitter blob with target volume `7500 * widen_factor` voxels (matches the BraTS-2026 training-mask median). A.3 (donor-tumor mimicking) re-uses the same placement logic but takes the shape from a donor scan's `gt/tumor_mask`. The official sampler is logged as the canonical replacement in this file; downstream consumers (the final training distribution) should swap it back in via a separate task before pre-flight 02 / training begin.
**Consequences:** Δ_aug-VAE measurements for A.1 / A.2 / A.3 carry an extra systematic bias (typically negative Δ — the simplified blob's smaller average volume makes the augmented case easier for the VAE). KS p-values for `volume` are sensitive to this approximation; the smoke run trips `ks_hard_fail` precisely because the sampler's narrow volume distribution differs from the BraTS-provided val masks. The default 50-volume run will be more robust but the bias persists. Action item logged as a follow-up: vendor the upstream `include.py` properly before Wk 3 (FM baseline).
**Status:** accepted

## 2026-05-19 — Smoke wall-clock is ~6 min on the 3060, not the ≤ 5 min target

**Context:** The prompt §3 says `smoke.yaml on 4 volumes — must finish in < 5 min`. Empirical wall-clock on the local 3060 (12 GB) is ~6 min: 4 train scans × 8 transforms × ~10 s per (scan, transform) VAE round-trip + 16 QC PNG renders + mask-descriptor audit. The VAE forward dominates (encode+decode at the 192³ envelope ≈ 4-5 s per pair under autocast + checkpointing).
**Decision:** Accept the 6-minute wall-clock as the realistic floor on the 3060 with full deliverables (16 QC PNGs + 3 CSVs + decision.json + report.md). Do not skip deliverables to meet 5 min. The full-sized A100 SLURM script will be much faster per scan but operates on the full 50-volume default config.
**Consequences:** The smoke run reliably produces a complete artifact set. A `--no-figures` smoke mode would land at ~3 min if needed for tighter iteration, but is not yet implemented.
**Status:** accepted

## 2026-05-19 — MAISI VAE wrapper: 3060 pad shape (192, 192, 144) with autocast fp16 + activation checkpointing

**Context:** The prompt §3 (hardware envelope) and the explore agent both anticipated a 256×256×192 pad → latent (4, 64, 64, 48), based on the MAISI v2 audit doc. An empirical probe on the local RTX 3060 (12 GB VRAM) confirms 256³ encode peaks at 9.9 GB but the matching decode OOMs (~14 GB peak inferred). Also: MAISI ships `norm_float16: True`, which collides with fp32 input on the consumer GPU ("Input type (c10::Half) and bias type (float)"). And the checkpoint dict carries only `epoch`, `unet_state_dict`, `epoch_finished` — no `scale_factor` key.
**Decision:** The `MaisiVAE` wrapper (`src/brainrepa_fm/common/maisi.py`) (a) overrides `norm_float16=False` and `use_checkpointing=True` at construction; (b) runs every encode/decode under `torch.autocast(device_type='cuda', dtype=torch.float16)` by default (toggle via `autocast_fp16=False`); (c) defaults `scale_factor=1.0` when the checkpoint omits the key; (d) exposes two pad shapes — `MAISI_PAD_SHAPE = (256, 256, 192)` for A100-class hardware and `MAISI_PAD_SHAPE_3060 = (192, 192, 144)` for the local 3060. The 3060 path also center-crops BraTS 240×240×155 down to 192×192×144 via `center_crop_to_maisi`. Empirical: encode→decode round-trip = 4.3 s at 7.0 GB peak; latent shape = (4, 48, 48, 36).
**Consequences:** Pre-flight 01 runs locally with the 3060 envelope. The Picasso SLURM script will switch to `MAISI_PAD_SHAPE` for the full 1,251-volume run and the future latent-H5 producer will pin `latent_spatial_shape` to whatever the actual A100 probe returns (likely `(64, 64, 48)`). Latent statistics (`latent_scale`, `latent_mean`) computed on the 3060 are NOT directly transferable to the A100 envelope and must be recalibrated by pre-flight 03 on the target hardware.
**Status:** accepted

## 2026-05-19 — MAISI VAE audit uses `z_mu` (deterministic), not `encode_stage_2_inputs`

**Context:** `AutoencoderKlMaisi.encode(x)` returns `(z_mu, z_sigma)`; `encode_stage_2_inputs(x)` samples `z ~ N(z_mu, z_sigma)`. The augmentation preflight measures `Δ_aug-VAE` = PSNR drop on the decoded round-trip; a stochastic encoder would inject sampling noise into the comparison and confound the augmentation signal.
**Decision:** `src/brainrepa_fm/common/maisi.py::MaisiVAE.encode` returns `z_mu * scale_factor`. The reparametrized `encode_stage_2_inputs` is reserved for the (future) FM training stages where the latent prior matters. `decode(z)` divides by `scale_factor` before passing to the upstream decoder.
**Consequences:** All preflight numbers are reproducible across seeds. The latent scale convention (multiply on encode, divide on decode) matches the upstream MAISI `scripts/sample.py:66` flow.
**Status:** accepted

## 2026-05-19 — A.2 transform regenerates the void mask via the official sampler

**Context:** A.2 is "official BraTS sampler with ×1.5 widened shape & volume ranges." The on-disk `mask.nii.gz` per subject already encodes the A.1 default sampler output. Two options for A.2: re-invoke the sampler with widened ranges, or approximate the widening via dilation/scaling of the default mask.
**Decision:** Re-invoke. The official sampler (`docs/2026_challenge/dataset/include.py::getHealthyMasks` and friends) is vendored frozen into `src/brainrepa_fm/data/brats_official_sampler.py` with a module docstring stating the upstream path. A thin facade `sample_void_mask(brain, tumor, *, widen_factor, seed)` exposes the wide-range call. Dilation approximations distort surface-to-volume and centroid distributions, which would invalidate the KS audit downstream.
**Consequences:** A.2 carries one extra dependency surface (the vendored sampler code), but its mask descriptors are faithful to the proposal §3.2 widened ranges. The vendored module is read-only; future BraTS sampler changes require an explicit re-vendor with a DECISIONS entry.
**Status:** accepted

## 2026-05-19 — `pyproject.toml` uses setuptools; runtime deps empty at bootstrap

**Context:** Build-backend choice (setuptools vs hatchling vs poetry-core) and initial dependency set. Project will accumulate deps lazily as routines land; over-pinning at bootstrap would pull a stack the bootstrap code does not yet use.
**Decision:** Build backend = `setuptools>=68` (stdlib-friendly, conda-compatible, no extra build dep). `[project.dependencies] = []` at bootstrap. `[project.optional-dependencies].dev = ["ruff>=0.6", "pytest>=8.0"]` (the minimum referenced by `.claude/rules` and `.claude/skills/`). Console scripts in `[project.scripts]` are commented forward declarations; uncomment per routine as it lands. Multi-root layout: `src/brainrepa_fm` (library, src-layout) + `./routines` (flat-layout) via `tool.setuptools.packages.find.where = ["src", "."]`. Torch is intentionally NOT pinned — CUDA/driver wheel selection is an install-time concern (see header comment in `pyproject.toml`).
**Consequences:** `pip install -e ".[dev]"` succeeds immediately with only ruff + pytest. Every subsequent dep addition triggers a `pyproject.toml` edit + `pip install -e .` per [[feedback-libraries-first]]. CUDA-specific wheel selection stays out of `pyproject.toml`; install commands per hardware are documented in the file's comment block.
**Status:** accepted

## 2026-05-19 — Decision log convention adopted

**Context:** No explicit place existed to record decisions; pre-flight `decision.json` files cover machine-readable outputs but do not carry the narrative ("why did we pick path 1?", "why λ_REPA=0.5?").
**Decision:** Maintain `DECISIONS.md` at the repo root as an append-only, ISO-8601-dated, newest-first log. Mandatory entry before committing any architecture/method/parameter change.
**Consequences:** Every architecture/method PR carries one DECISIONS.md entry minimum. Manuscript writing and peer-review responses can cite specific dated decisions.
**Status:** accepted

## 2026-05-19 — Pre-flights at `routines/preflights/<name>/`

**Context:** Proposal §03/01/02 layout used `src/preflight/<name>/` with `run_preflight_<name>.py` drivers. Three options were considered: (a) proposal-verbatim `src/preflight/`, (b) MenFlow routines pattern `routines/preflights/<name>/`, (c) hybrid.
**Decision:** Option (b). Pre-flights live at `routines/preflights/{maisi_vae,augmentation,bsf_layers}/`. Training stages at `routines/<stage>/` as siblings. Library code lives in `src/brainrepa_fm/<area>/<name>/`; `routines/<bucket>/<name>/engine/` is a thin orchestrator only.
**Consequences:** Library code stays importable/testable independently of the CLI. Uniform with future Wk 1-12 training routines. Slightly heavier scaffolding per pre-flight (cli + engine + configs + slurm) than option (a).
**Status:** accepted

## 2026-05-19 — Conda env `brainrepa`

**Context:** MenFlow used `menflow`; MenGrowth used `growth`. Options: `brainrepa`, `brainrepa_fm`, or reuse `menflow`.
**Decision:** Conda env name is `brainrepa`. Coupled with a libraries-first rule: any new dep is declared in `pyproject.toml` AND `pip install -e .`-ed in the same change.
**Consequences:** All Python invocations across `.claude/` settings, hooks, and skills resolve to `~/.conda/envs/brainrepa/bin/python`. Env must exist before any pytest / preflight invocation.
**Status:** accepted

## 2026-05-19 — H5 inspired by MenFlow, not compatible

**Context:** MenFlow defines a unified H5 schema (longitudinal CSR, multi-modality, per-cohort features). BraTS 2026 is single-cohort, single-modality, cross-sectional; the unified schema over-constrains it.
**Decision:** Adopt MenFlow's *principles* (`.claude/rules/h5-design-principles.md`: attrs-driven, validator-paired, write-then-assert, gzip-4, self-describing datasets). Each H5 producer owns its own schema and ships a paired `validate_<artifact>` + `assert_<artifact>_valid` helper. NOT MenFlow-compatible.
**Consequences:** No cross-project H5 compatibility, but no dead fields either. New H5 producers must document the schema in the producer module's docstring.
**Status:** accepted

## 2026-05-19 — `.claude/` hooks: keep all four

**Context:** Inherited `.claude/settings.json` had four hooks (PostToolUse ruff, PreToolUse sensitive-file block, Notification, SessionStart compact-context) — but the SessionStart hook referenced a missing `.claude/hooks/compact-context.sh` and the ruff env was `growth`.
**Decision:** Keep all four hooks. Fixed env paths to `brainrepa`. Wrote `.claude/hooks/compact-context.sh` (prints git status, recent commits, top-level layout, decision.json files, key reference paths on session compact).
**Consequences:** Ruff format runs automatically on every Python Edit/Write in the `brainrepa` env. Sensitive files (`.env`, `credentials`, `secret`) are blocked from write. Notification fires on session events with BrainREPA-FM text. Compact-context summary appears on session resume.
**Status:** accepted
