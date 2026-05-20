# BrainREPA-FM — Project Hub

**Project:** Submission to the ASNR-MICCAI BraTS-Lighthouse 2026 inpainting challenge.

**Task:** Given a T1-weighted brain MRI with a void mask `m_v`, predict the healthy tissue under the void.

**Method:** 3D latent flow-matching generator (DiT / U-Net with attention) in the MAISI-V2 VAE latent space, aligned to frozen BrainSegFounder-Small features via REPA, with a pixel-space `L1 + (1-SSIM)` head on the decoded volume. Final output is feathered paste-back: `(1-m_v) ⊙ x̃ + m_v ⊙ x̂` with 2-3 voxel feathering at the mask boundary.

This file is the **hub of paths and conventions**. Detailed rules live in `.claude/rules/`. Detailed specs live in the external docs folder.

---

## Documentation (source of truth)

| What | Where |
|---|---|
| Full proposal (method, loss, training stages, evaluation) | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/proposal.md` |
| Pre-flight check index & dependency graph | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/checks/00_README.md` |
| Pre-flight 01 (augmentation) spec | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/checks/01_augmentation_preflight.md` |
| Pre-flight 02 (BSF layer selection) spec | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/checks/02_bsf_layer_analysis.md` |
| Pre-flight 03 (MAISI VAE audit) spec | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/checks/03_maisi_vae_audit.md` |

## External dependencies (mirrored from `src/external/README.md`)

| Asset | Path |
|---|---|
| BrainSegFounder source | `/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/code/BrainSegFounder` |
| BrainSegFounder SSL checkpoint (UK Biobank) | `/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt` |
| MAISI-V2 source | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR` |
| MAISI-V2 VAE-GAN checkpoint | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI-V2 FM checkpoint (reference) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt` |
| BraTS 2026 training (NIfTI) | `/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training` |
| BraTS 2026 validation (no GT) | `/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation` |
| BraTS 2026 H5 cache | `/media/mpascual/MeningD2/INPAINTING/2026/h5/brats_inpainting_2026.h5` |

If this table drifts from `src/external/README.md`, that file wins.

---

## Dataset — BraTS-Inpainting 2026 (Local Synthesis)

Glioma-only cohort, T1-weighted, cross-sectional. Derived from BraTS-2023-GLI training data; every case ID is `BraTS-GLI-{id}-{tp}`. There is **no meningioma** in the training or validation data — cross-pathology (GLI vs MEN) is a Week-8 generalization concern, not a property of this dataset.

**Per training case** — 2 image volumes + 3 binary masks:

| File `{type}` | Meaning |
|---|---|
| `t1n` | Original / ground-truth T1: full brain, healthy tissue **and** glioma present, nothing voided. |
| `t1n-voided` | Input T1 with the void region set to background (zero). Two zones are voided: the glioma, and a randomly selected healthy region. |
| `mask` | Binary mask of **all** voided voxels (`= mask-healthy ∪ mask-unhealthy`). The region to inpaint. |
| `mask-healthy` | Subset of `mask` over healthy tissue. Healthy GT is known here → used for supervised training/eval. |
| `mask-unhealthy` | Subset of `mask` over the glioma / unhealthy tissue (slightly enlarged tumor region). |

**Per validation (`challenge_val`) case** — voided volume + void mask only: `t1n-voided` and `mask`. **No** `t1n` (GT), `mask-healthy`, or `mask-unhealthy`. An algorithm must run on `t1n-voided` + `mask` alone.

**Evaluation** (Synapse server) — SSIM, PSNR, MSE, computed **only inside `mask`**. No pathology-classification metric.

H5 cache `brats_inpainting_2026.h5` (schema owned by `src/brainrepa_fm/data/brats2026_converter.py`, validated by `assert_brats2026_valid`) holds all 1,470 scans; `split ∈ {train (1006), val (117), test (128), challenge_val (219)}`. Field mapping: `t1n-voided → images/t1_voided`, `mask → masks/void` (all rows); `t1n → gt/t1`, `mask-healthy → gt/healthy_mask`, `mask-unhealthy → gt/tumor_mask` (sparse `gt/` group, training-pool rows only, indexed by `gt/scan_index`). Challenge tutorial + dataset-generation algorithm: `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/2026_challenge/`.

---

## Project structure (target layout)

```
BrainREPA-FM/
├── CLAUDE.md                       # this file
├── pyproject.toml                  # deps; console scripts brainrepa-<bucket>-<name>
├── src/
│   ├── brainrepa_fm/               # library code (importable, unit-testable)
│   │   ├── common/maisi.py         # shared VAE primitives
│   │   ├── preflight/              # library implementations of pre-flights
│   │   │   ├── augmentation/{transforms,vae_composability,mask_audit,visualize}.py
│   │   │   ├── bsf/{adapt_t1,cknna,linear_probes}.py
│   │   │   └── maisi_vae/{reconstruction,equivariance}.py
│   │   ├── adapters/bsf_t1.py      # T1-only patch-embed wrapper
│   │   └── data/                   # cohort loaders, H5 producers
│   └── external/README.md          # canonical external paths (read-only otherwise)
├── routines/                       # CLI entrypoints (one YAML arg, thin engine wrappers)
│   ├── preflights/{augmentation,bsf_layers,maisi_vae}/
│   └── <stage>/                    # Wk 1-12 training/eval routines
├── artifacts/<routine>/<UTC>/      # report.md, figures/, tables/, decision.json
├── tests/                          # pytest
└── .claude/                        # agentic rules + skills
```

## Pre-flight gates

Stage-2 FM training is gated on three pre-flights. Order: **03 → (01 ∥ 02) → FM training**. Wk 1-2 baseline U-Net is independent.

| Routine | Decision keys (sourced from `docs/checks/`) |
|---|---|
| `routines/preflights/maisi_vae` | `path` (1=frozen / 2=fine-tune / 3=wavelet), `vae_fine_tune`, `latent_aug_safe`, `latent_scale` |
| `routines/preflights/augmentation` | `include`, `drop`, `halve_range`, `ks_hard_fail` |
| `routines/preflights/bsf_layers` | `feasible`, `ell_star` (0-4), `stage_shape_at_ell_star`, `resample_mode`, `checkpoint_path` |

Each writes to `artifacts/<routine>/<UTC-timestamp>/decision.json`. Downstream training routines load this file at startup; they never re-derive its conclusions.

---

## Decision log (mandatory)

Every non-trivial decision (architecture choice, hyperparameter pin, library swap, scope change, deviation from the proposal) **must** be appended to `DECISIONS.md` at the repo root **before** the code change is committed. Format (newest entry on top):

```
## YYYY-MM-DD — <short title>

**Context:** <what was the question / what alternatives existed>
**Decision:** <what we picked>
**Consequences:** <what changes downstream, what we lose, what we gain>
**Status:** accepted | superseded by <YYYY-MM-DD-other-title>
```

Rules:

- ISO-8601 date prefix on every entry. No relative dates ("yesterday", "last week") — they decay.
- Status updates *edit* the original entry's status line. Never delete entries.
- Pre-flight `decision.json` files are the machine-readable counterpart consumed by code. `DECISIONS.md` is the human-readable narrative consulted during manuscript writing and peer-review responses.
- Library swaps count as decisions (see `.claude/rules/coding-standards.md` item 6). H5 schema picks count as decisions (see `.claude/rules/h5-design-principles.md`).

---

## Conventions index (`.claude/rules/`)

| Rule | Subject |
|---|---|
| `coding-standards.md` | Python style, types, logging, env, 3D, frozen-models, libraries-first |
| `preflight-pattern.md` | Routine layout, `decision.json` schemas, validate-on-close |
| `h5-design-principles.md` | H5 producer rules (attrs, validators, storage policy) |
| `external-deps.md` | Frozen-model hard rules, adapter location, dataset path policy |
| `training-stages.md` | Wk 1-12 dependency graph, routine name conventions |

## Skills index (`.claude/skills/`)

| Skill | When to invoke |
|---|---|
| `/explore` | Codebase exploration scoped to BrainREPA-FM |
| `/test` | Run pytest with project conda env and marker shortcuts |
| `/refactor` | Refactor a Python module against the production checklist |
| `/dl-scientist` | Analyze training results with project-specific citations |

## Subagent routing (project-specific overrides)

The global routing rules from `~/.claude/CLAUDE.md` apply. Project-specific overrides:

- Pre-flight runs that ingest `docs/proposal.md` or `docs/checks/*` go through the `read-paper` subagent (Sonnet 1M) — the proposal is too long for the main thread.
- VAE / BSF result audits go through `dl-scientist` first, then `research-rigor` (Opus) when proposing a method change.
- SLURM script generation invokes the `picasso-sbatch` skill or `slurm-builder` subagent. Picasso constraints: Singularity-only, `--constraint=dgx`, A100-40GB, no `module load python`.

## Quick commands

```bash
# Env
~/.conda/envs/brainrepa/bin/python --version

# Tests (safe default)
~/.conda/envs/brainrepa/bin/python -m pytest tests/ -v -m "not slow and not gpu"

# Run a pre-flight (once routines exist)
~/.conda/envs/brainrepa/bin/python -m routines.preflights.maisi_vae.cli \
    routines/preflights/maisi_vae/configs/default.yaml

# Install / sync deps after editing pyproject.toml
~/.conda/envs/brainrepa/bin/pip install -e .

# Lint + format
~/.conda/envs/brainrepa/bin/python -m ruff check src/ tests/
~/.conda/envs/brainrepa/bin/python -m ruff format src/ tests/
```

## Out-of-scope reminders

- No 2D ops in the core pipeline. 2.5D only for LPIPS aggregation in eval.
- Void fill is exactly zero. No non-zero fill augmentation.
- Frozen models stay frozen. Never edit `src/external/`.
- One H5 producer owns one H5 schema; the principles in `.claude/rules/h5-design-principles.md` are non-negotiable.
- New dependencies go through `pyproject.toml` + `pip install -e .` in the `brainrepa` env in the same change.
