---
name: explore
description: Deep codebase exploration for BrainREPA-FM (T1 inpainting, BraTS 2026, latent FM + REPA)
---

# BrainREPA-FM Codebase Exploration

Thoroughly explore the BrainREPA-FM codebase to answer the query.

## Project at a glance

BrainREPA-FM is the proposal for the **ASNR-MICCAI BraTS-Lighthouse 2026 inpainting** challenge: given a T1 brain MRI with a void mask `m_v`, predict the healthy tissue under the void. Method: **3D latent flow-matching generator** (DiT or U-Net with attention) operating in the MAISI-V2 VAE latent space, with **REPA-style alignment** to frozen BrainSegFounder-Small (SwinUNETR) features, plus a pixel-space `L1 + (1-SSIM)` head on the decoded volume. Final output: feathered paste-back `(1-m_v) ⊙ x̃ + m_v ⊙ x̂`.

| Property | Value |
|---|---|
| Modality | T1 only (single channel) |
| Native shape | `(240, 240, 155)` |
| VAE input shape | `(256, 256, 192)` (padded, divisible by 4) |
| Latent shape | `(64, 64, 48, 4)` (4× spatial compression, 4 channels) |
| Train cohort | BraTS-GLI (1,251 volumes) |
| Val cohort | BraTS-GLI (219 volumes, no GT) |
| Test cohort | BraTS-GLI + BraTS-MEN (hidden) |

## Pre-flight gates (must inspect before assuming any architecture decision)

Three pre-flight checks gate Stage-2 FM training. Each writes a `decision.json` under `artifacts/preflights/<name>/LATEST/`. **When the query touches architecture, pretrained models, augmentations, or training stages, read the relevant `decision.json` first** — do not infer from code or docs alone.

| Check | Spec | Decision keys |
|---|---|---|
| `preflights/maisi_vae` | `docs/checks/03_maisi_vae_audit.md` | `path` (1/2/3), `vae_fine_tune`, `latent_aug_safe`, `latent_scale` |
| `preflights/augmentation` | `docs/checks/01_augmentation_preflight.md` | `include`, `drop`, `halve_range`, `ks_hard_fail` |
| `preflights/bsf_layers` | `docs/checks/02_bsf_layer_analysis.md` | `ell_star` (1-4), `adaptation_mode`, `checkpoint_path` |

Dependency: **03 → (01 ∥ 02) → Stage-2 FM training**. Wk 1-2 baseline U-Net is independent and may run in parallel.

## Key locations

- `CLAUDE.md` (repo root) — hub of paths, conventions, quick commands
- `src/external/README.md` — canonical external dependency paths (MAISI, BSF, datasets)
- `src/brainrepa_fm/` — library code; importable, unit-testable
  - `common/maisi.py` — shared MAISI encode/decode primitives (used by `preflight/augmentation/` and `preflight/maisi_vae/`)
  - `preflight/{augmentation,bsf,maisi_vae}/` — library implementations of each pre-flight
  - `adapters/bsf_t1.py` — T1-only patch-embed rewrite around BSF-S
- `routines/` — CLI entrypoints (one YAML arg, thin engine wrappers)
  - `routines/preflights/{augmentation,bsf_layers,maisi_vae}/`
  - `routines/<stage>/` for Wk 1-12 (see `.claude/rules/training-stages.md`)
- `artifacts/<routine>/<UTC-timestamp>/` — outputs: `report.md`, `figures/`, `tables/`, `decision.json`
- `tests/` — pytest. Markers: `unit`, `preflight_maisi`, `preflight_bsf`, `preflight_aug`, `fm`, `repa`, `gpu`, `slow`

## Documentation source-of-truth

External docs at `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/`:

- `proposal.md` — full method, loss formulation, training stages, evaluation
- `checks/00_README.md` — pre-flight index and dependency graph
- `checks/0{1,2,3}_*.md` — per-check specs (acceptance criteria, decision rules, deliverables)

## Critical conventions

- **3D throughout** — no 2D ops in the core pipeline. 2.5D LPIPS is the only exception, used for perceptual eval.
- **Frozen models** (MAISI VAE, BSF-S) are never written to. Adapter wrappers go in `src/brainrepa_fm/adapters/`.
- **No unified H5 schema.** Each H5 artifact owns its layout but satisfies the principles in `.claude/rules/h5-design-principles.md`.
- **Conda env:** `brainrepa`. Pytest: `~/.conda/envs/brainrepa/bin/python -m pytest tests/ -v`.
- **Library code lives in `src/brainrepa_fm/`**, routines are thin wrappers. Never put implementation logic inside `routines/<name>/engine/`.

## Loss components (cheat sheet)

Total: `L = L_FM + λ_REPA · L_REPA + L_pix` with `λ_REPA ∈ {0.25, 0.5, 1.0}` (swept in Wk 5-6).

- `L_FM` — linear-interpolant flow matching on latents; generator inputs `[z_t, ẑ, m̂_v]` concatenated channel-wise.
- `L_REPA = 1 - cos(h_φ(g_ℓ*(·)), BSF_ℓ*(·))` — cosine alignment of generator hidden state at layer ℓ* to BSF feature at corresponding stage. `h_φ` is an MLP discarded at inference.
- `L_pix = λ_1 · ‖m_v ⊙ (x̂ - x)‖_1 + λ_2 · (1 - SSIM)` on decoded volume; `λ_1 = 1, λ_2 = 0.5`.

$ARGUMENTS
