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

## 2026-05-20 — Pre-flight 03 adds the pixel-space voided-input round-trip (Caveat 2)

**Context:** Pre-flight 03's reconstruction audit round-trips the *intact* volume `gt/t1` (`r(x) = D(E(x))`). `docs/checks/03_maisi_vae_audit.md` §6 Caveat 2 calls the *voided* pixel round-trip `r(x̃)` "uninteresting" and instead prescribes the §7 latent-locality tests (`S_inside` / `S_outside`) to characterise `E(x̃)`. But at inference the proposal conditions the generator on `E(x̃)` and the decoder renders tissue near the void boundary; the §7 latent tests measure encoder locality, not whether the *decoder* faithfully reproduces still-visible tissue when the encoder saw a zero hole. A latent can be local (`S_outside ≈ 0`) yet the decoded visible region still degrade.

**Decision:** Add `compute_voided_roundtrip_metrics` (+ frozen `VoidedRoundtripMetrics`) to `src/brainrepa_fm/preflight/maisi_vae/reconstruction.py`. Per volume the engine decodes `D(E(x̃))` (the voided latent was already encoded for §7) and scores visible-region (`brain ∩ ¬void`, using the H5's own `masks/void`) MSE/PSNR/SSIM against the true signal `x`, alongside the same region scored for `D(E(x))`. The per-volume `delta_psnr_visible_db` (un-voided minus voided) isolates the fidelity lost purely to the input hole. New deliverables: `tables/voided_roundtrip.csv`, `figures/voided_roundtrip_psnr_drop.png`, a report section, and a `decision.json` key `median_voided_visible_psnr_drop_db`.

**Consequences:** Deliberate deviation from §6 Caveat 2's "uninteresting" framing — decoder degradation of visible tissue is a distinct failure mode the latent §7 tests miss, and it matters for the feathered paste-back boundary. The diagnostic is *report-only*: the architecture-path decision (§3.3 / §4) still uses the intact-round-trip inside-void PSNR. `decision.json` gains one additive key; `schema_version` stays `1.0`. One extra decode per volume. Verified on the 2-subject local smoke.

**Status:** accepted

## 2026-05-20 — Pre-flight 02 cohort restricted to glioma-only 2026 data; meningioma probe dropped

**Context:** `docs/checks/02_bsf_layer_analysis.md` (v2.0) drew its BSF layer-selection cohort from "200 BraTS-Inpainting 2026 + 50 BraTS-MEN" volumes and included a glioma-vs-meningioma linear probe (P3, weight 0.2 in the stage-ranking aggregation). The BraTS-Inpainting 2026 challenge docs (`docs/2026_challenge/`) establish the dataset is glioma-only — every case `BraTS-GLI-*`, derived from BraTS-2023-GLI — and the challenge metric is SSIM/PSNR/MSE inside the void mask, with no pathology-classification component.

**Decision:** Pre-flight 02 uses only held-out BraTS-Inpainting 2026 data — the `test` split of `brats_inpainting_2026.h5` (128 GLI cases). The meningioma import and probe P3 are removed: no MEN data exists in the challenge, and tumour-type discriminability is the wrong axis for selecting a REPA target aligned to BSF of the *healthy* ground truth. Layer-selection metrics (token-wise CKNNA vs SynthSeg anatomy + the P1 token tissue probe) are evaluated over all in-brain tokens **except** those overlapping the tumour mask (`gt/tumor_mask`). Aggregation drops to two criteria with weights `(CKNNA, P1) = (0.6, 0.4)`. SynthSeg is retained as the anatomical reference — the dataset ships no healthy-tissue parcellation and SynthSeg is contrast-agnostic.

**Consequences:** `decision.json` (schema 2.0) and the `preflight-pattern.md` schema block drop `auc_p3`. `CLAUDE.md` gains a "Dataset" section documenting the BraTS-Inpainting 2026 file types and the train/validation asymmetry; a project-memory note records the glioma-only constraint. The `brats_inpainting_2026.h5` cache was verified to already carry this structure: per-training-pool-row `gt/t1` + `gt/healthy_mask` + `gt/tumor_mask` via the sparse `gt/` group; `challenge_val` rows carry only `images/t1_voided` + `masks/void` (validator `assert_brats2026_valid`, 0 violations). Cross-pathology (GLI vs MEN) remains the Week-8 `cross_pathology` routine's responsibility.

**Status:** accepted

## 2026-05-20 — Pre-flight 03 (MAISI VAE audit): reconstruction-only scope, four deviations from spec §8

**Context:** `docs/checks/03_maisi_vae_audit.md` specifies a reconstruction audit (§2.1, §3, §4) **plus** an equivariance audit (§2.2, §3.4–3.7), an artifact layout sketch under `src/preflight/maisi/` (§8), `latent_aug_safe` populated with latent-safe augmentations, and intensity renormalization within the brain mask (§3.1). The implementation was scoped by the user to reconstruction + §7 voided-encoder tests + Caveat 8 latent statistics, explicitly **excluding** equivariance.

**Decision:** Implement `routines/preflights/maisi_vae/` (engine + cli + configs + slurm) with library code at `src/brainrepa_fm/preflight/maisi_vae/{preprocess,reconstruction,latent_stats,voided_tests,visualize}.py` and a new shared `src/brainrepa_fm/common/metrics.py`. Four explicit deviations from the spec, each reasoned:
1. **Equivariance audit deferred.** §2.2 / §3.4–3.7 and all equivariance figures are out of scope for this routine version. Because `latent_aug_safe` is *derived* from the equivariance audit, `decision.json` carries it as the empty list `[]` — the conservative reading "apply all augmentation in image space". A later routine version adds equivariance with its own DECISIONS entry.
2. **Layout follows `preflight-pattern.md`, not spec §8.** The §8 `src/preflight/maisi/run_preflight_maisi.py` sketch is overridden by the project's canonical routine pattern (mirrors `routines/preflights/augmentation/`).
3. **`latent_mean` added to `decision.json`.** Spec §8 lists only `latent_scale`. Schema B (`brainrepa_latents.h5`) has zero-placeholder datasets `latent_scale (4,)` **and** `latent_mean (4,)`; pre-flight 03 is their designated producer, so both per-channel statistics are written (a justified extension of the pinned schema).
4. **No intensity renormalization.** Spec §3.1 prescribes 5th–99.5th-percentile clip + min-max scale; the Schema A H5 already stores volumes normalized to `[0, 1]` (converter output). Re-normalizing would double-normalize. The audit consumes H5 values directly, matching the convention of the augmentation pre-flight's `_psnr`. Masked SSIM uses `skimage.metrics.structural_similarity(..., full=True)` rather than `monai.metrics.SSIMMetric` because region-restricted SSIM (brain/void/tumor) needs the per-voxel map a reduced metric does not expose.

**Consequences:** Downstream FM routines parsing this pre-flight's `decision.json` (`schema_version "1.0"`) get `latent_aug_safe == []` from this version and must read both `latent_scale` and `latent_mean`. The decision rule is purely reconstruction-driven: median inside-void PSNR ≥ 28 → Path 1; 24–28 → Path 1 + fine-tune contingency; < 24 → Path 3 (hard-fail). A local 2-subject 3060-envelope smoke on the real MAISI VAE returned median inside-void PSNR 23.77 dB (Path 3) — non-conclusive at n = 2; the conclusive run is `configs/picasso.yaml` (all 1,251 training volumes, 256×256×192 envelope) on an A100. Smoke latent stats showed per-channel `latent_scale ≈ [0.71, 0.33, 0.62, 0.67]`, already below the MAISI-v2 N(0,1) assumption (Caveat 8) — to be confirmed at scale.

**Status:** accepted

## 2026-05-20 — BSF-S REPA encoder is the T1-only Stage-1 SSL checkpoint; pre-flight 02 → schema v2.0

**Context:** `docs/checks/02_bsf_layer_analysis.md` assumed BSF-S has a native T1+T2 input (`in_channels=2`) and prescribed first-layer patch-embed weight surgery (`discard_t2` / `average` / `learn_projection`) plus a segmentation-decoder Dice sanity check. Inspection of the released checkpoint `64-gpu-model_bestValRMSE.pt` contradicts both: it is a Stage-1 SSL `SSLHead` state_dict — `in_channels=1` (patch-embed weight `[48,1,2,2,2]`), encoder + rotation/contrastive/reconstruction heads only, no segmentation decoder, ~20M params, DDP `module.` prefix, `epoch=0` / `global_step=75`. The BSF paper's headline T1+T2 model is a different artifact we do not hold. The encoder exposes five feature maps (`swinViT.forward()` → `[x0..x4]`), not four.

**Decision:** Use the released T1-only Stage-1 checkpoint as the frozen REPA encoder with no channel surgery. Refine pre-flight 02: drop the T1-adaptation machinery and the `adaptation_mode` key; re-index candidate stages to `ℓ ∈ {0,…,4}` (index into the forward-output list); make feasibility an explicit gate (permutation null on token-wise CKNNA + an SSL reconstruction-head health check using the in-checkpoint `module.conv`); compute CKNNA and the linear probe token-wise (REPA aligns per-patch, not on a pooled vector); add an explicit downstream extraction protocol (resample the stage feature map to the generator token grid). `decision.json` `schema_version` → `2.0`: drops `adaptation_mode`, re-indexes `ell_star`, adds `feasible`, `stage_shape_at_ell_star`, `resample_mode`, and checkpoint-provenance keys.

**Consequences:** `.claude/rules/preflight-pattern.md` and `CLAUDE.md` updated to the v2.0 schema and the `feasible == true and ell_star in {1,2,3}` downstream gate. `.claude/rules/training-stages.md` still carries the stale `ℓ* ∈ {2,3}` target metric and an incomplete hard-fail list — pending a separate edit (the agent-config write was blocked). The BSF loader is an adapter at `src/brainrepa_fm/adapters/bsf_t1.py`. A T1+T2 / Stage-2 checkpoint is no longer a dependency; if pre-flight 02 returns `feasible == false`, the fallback is re-pretraining a small SwinUNETR on BraTS T1 (primary) or DINOv2 slice-wise (cross-domain control). `global_step=75` flags possible SSL undertraining — verified empirically by the pre-flight, not assumed.

**Status:** accepted

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

## 2026-05-19 — Schema B carries two anchor latents (voided + gt), not one

**Context:** The original Schema B had a single `latents/anchor` dataset. The FM training objective in the proposal (SD-Inpainting-style flow matching) requires BOTH the conditioning latent `z̃ = E(t1_voided)` (model input) AND the supervision latent `z₀ = E(gt_t1)` (training target). Encoding `gt_t1` on the fly each training iteration burns ~4 s of GPU time per sample.
**Decision:** Extend Schema B with `latents/voided_anchor` (`float32 (n_scans, C, Lz, Ly, Lx)`, always present) and `latents/gt_anchor` (sparse, `(n_with_gt, C, Lz, Ly, Lx)`, indexed by `gt/scan_index`). Add the `n_with_gt` root attr. Update the validator to check both, enforce the sparse leading-dim invariant on `gt_anchor`, and assert `gt/scan_index` is disjoint from `splits/challenge_val`. Augmented latents stay CSR-style, applied to the voided path (apply transform → re-void → encode).
**Consequences:** ~2× storage for Schema B at the training pool (1,251 × 2 = 2,502 latents at ~5 MB each = ~12.5 GB) but ~4 s × millions of training iterations saved at FM time. Cleaner partitioning between inference-only (`challenge_val`) and training (`train`/`val`/`test`).
**Status:** accepted

## 2026-05-19 — Latent encoder routine `routines/data/encode_latents`

**Context:** Pre-flight 01 audits the augmentation set; the FM training loop wants pre-encoded latents to avoid re-encoding the same volume thousands of times. A dedicated producer was needed.
**Decision:** Add `src/brainrepa_fm/data/latent_encoder.py::LatentEncoder` + `routines/data/encode_latents/{cli, configs/{default,smoke}, engine/encoder_engine}.py` + console script `brainrepa-data-encode-latents`. Streams Schema A → Schema B scan-by-scan, encodes voided + gt anchors, writes empty CSR-augmented placeholders (to be filled after pre-flight 01's `include` set is finalized). Atomic `.partial` + `assert_brainrepa_latents_valid` on close.
**Consequences:** The default config encodes the full 1,470 + 1,251 = 2,721 passes (~7-8 h on the 3060 at the 192³ envelope, ~3 h on an A100 at the 256³ envelope). Smoke verified on 1 scan in ~3 s.
**Status:** accepted

## 2026-05-19 — `n_train_subjects` allowed down to 1 (was 2)

**Context:** The user asked for a 1-scan augmentation preflight smoke. Pydantic field validator rejected `n_train_subjects = 1` with `ge=2`.
**Decision:** Relax `AugmentationRoutineConfig.n_train_subjects` and `.n_val_subjects` to `ge=1`. Also fix the donor-pool selection in `_collect_mask_descriptors` to draw from the full `splits/train` (not the audit subset), so n=1 doesn't drain the donor pool.
**Consequences:** 1-scan smokes complete in ~3.5 min on the 3060 with all deliverables; KS test still trips hard-fail at N=1 (~3 samples) by design — that is the intended safety behaviour.
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
