# Training Stages (Wk 1–12)

The 12-week training plan is specified in `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/proposal.md`. This rule pins the dependency order so agents do not skip stages or invent stage names.

## Dependency graph

```
[ Pre-flights ]                                                                                         
   maisi_vae  ────┐                                                                                     
   augmentation ──┼──► baseline_unet  (Wk 1-2, parallel to pre-flights)                                 
   bsf_layers ───┘                                                                                      
                  │                                                                                     
                  ▼                                                                                     
         fm_baseline  (Wk 3-4)                                                                          
                  │                                                                                     
                  ▼                                                                                     
         fm_repa      (Wk 5-6,  +L_REPA, λ sweep)                                                       
                  │                                                                                     
                  ▼                                                                                     
         fm_repa_pixel (Wk 7,   +L_pix on decoded volume)                                               
                  │                                                                                     
                  ▼                                                                                     
         cross_pathology (Wk 8, GLI vs MEN per-cohort scores)                                           
                  │                                                                                     
                  ▼                                                                                     
         downstream_eval (Wk 9, FastSurfer-LIT Dice + cortical thickness)                               
                  │                                                                                     
                  ▼                                                                                     
         uq             (Wk 10, voxelwise std N=10, conformal α=0.1)                                    
                  │                                                                                     
                  ▼                                                                                     
         final_ablations (Wk 11, TTA / augs / REPA-off / BSF→random / BSF→DINOv2)                       
                  │                                                                                     
                  ▼                                                                                     
         submit         (Wk 12, MLCube containerization + Synapse submission)                           
```

## Routine names (canonical)

Each stage maps to one routine under `routines/<name>/`, following the pattern in `preflight-pattern.md`:

| Wk | Routine | Purpose | Target metric |
|---|---|---|---|
| pre | `routines/preflights/maisi_vae` | Proposal §03 — VAE audit (highest-priority gate) | path ∈ {1,2,3} decision |
| pre | `routines/preflights/augmentation` | Proposal §01 — VAE composability + KS audit | accept set of 7-8 transforms |
| pre | `routines/preflights/bsf_layers` | Proposal §02 — BSF feasibility + REPA target ℓ* selection | `feasible` + ℓ* ∈ {1,2,3} |
| 1-2 | `routines/baseline_unet` | Reproduce Zhang 2024 U-Net | SSIM ≥ 0.84, PSNR ≈ 23 dB |
| 3-4 | `routines/fm_baseline` | Latent FM only (no BSF, no L_pix) | Isolates FM contribution |
| 5-6 | `routines/fm_repa` | Add L_REPA; sweep λ_REPA ∈ {0.25, 0.5, 1.0} | Best ℓ*-λ trade-off |
| 7 | `routines/fm_repa_pixel` | Add L_pix on decoded volume | Beats baseline on MSE/PSNR/SSIM |
| 8 | `routines/cross_pathology` | Per-cohort scores (GLI vs MEN) | Δ scores < proposal threshold |
| 9 | `routines/downstream_eval` | FastSurfer-LIT parcellation Dice + cortical thickness | Reproducibility vs reference |
| 10 | `routines/uq` | Voxelwise std (N=10), conformal calibration | Coverage ≈ 1-α at α=0.1 |
| 11 | `routines/final_ablations` | TTA, augs, REPA-off, BSF→random, BSF→DINOv2 | 7-row ablation table |
| 12 | `routines/submit` | MLCube containerization + Synapse submission | Successful leaderboard upload |

## Dependency rules

1. **Pre-flights gate all FM/REPA work.** `fm_baseline` and downstream routines refuse to start if any of the three pre-flight `decision.json` files is missing or carries a hard-fail flag (e.g. `ks_hard_fail: true`, `feasible == false` for BSF, `path == "3"` without explicit Path-3 routing).
2. **Wk 1-2 can run in parallel to pre-flights.** `baseline_unet` reproduces Zhang 2024 directly in pixel space; it does not depend on MAISI, BSF, or the latent path.
3. **One Wk N routine per directory.** Variants (e.g. `λ_REPA = 0.5` vs `λ_REPA = 1.0`) are separate YAML configs under the same routine — `routines/fm_repa/configs/lambda_0_5.yaml`, `lambda_1_0.yaml`. Not separate routines.
4. **Ablation table is built from `final_ablations`.** Each row corresponds to one YAML config under `routines/final_ablations/configs/`. The seven baselines listed in the proposal (Zhang U-Net reproduced, Latent FM only, FM+REPA, Full system, Full REPA-off, BSF→random SwinUNETR, BSF→DINOv2) are the minimum set.
5. **Picasso A100-40GB sizing.** Full-system training (~150k iter on 1,251 volumes) targets ~5-6 days on a single A100; VAE audit ~10 h; BSF analysis ~8 h; augmentation pre-flight ~4 h. SLURM scripts must request these budgets explicitly via `--time` and `--mem`.

## How to add a new routine (procedure)

1. Read `docs/proposal.md` for the corresponding week, plus the relevant pre-flight `decision.json` files.
2. Copy `routines/preflights/augmentation/` (or whichever existing routine is closest) as a starting skeleton.
3. Implement library code under `src/brainrepa_fm/<area>/`, never inside `routines/`.
4. Register the console script in `pyproject.toml`.
5. Write a smoke YAML (`configs/smoke.yaml`) that runs end-to-end on a 4-volume subset in < 5 minutes.
6. Add a pytest under `tests/<area>/test_<name>_engine.py` (mock the model for unit; mark `slow` for GPU smoke).
