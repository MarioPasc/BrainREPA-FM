---
name: test
description: Run pytest tests for the BrainREPA-FM project
---

Run the following tests and report results:

`~/.conda/envs/brainrepa/bin/python -m pytest $ARGUMENTS -v --tb=short`

If `$ARGUMENTS` is empty, run the safe default (excludes slow and GPU tests):

`~/.conda/envs/brainrepa/bin/python -m pytest -m "not slow and not gpu" -v --tb=short`

## Marker shortcuts

| Marker | Meaning |
|---|---|
| `unit` | Fast, no I/O, no GPU |
| `preflight_maisi` | MAISI VAE audit components (reconstruction, equivariance, latent stats) |
| `preflight_bsf` | BrainSegFounder layer analysis (CKNNA, linear probes, T1 adaptation) |
| `preflight_aug` | Augmentation pipeline (transforms, VAE composability, KS mask audit) |
| `fm` | Flow-matching training/inference components |
| `repa` | REPA loss / projection-head components |
| `gpu` | Requires CUDA |
| `slow` | Wall-clock > 30 s |

Combine with: `-m "preflight_maisi and not slow"`, `-m "fm and gpu"`, `-m "unit and not gpu"`.

## Report format

Report only:
- Total tests collected
- Pass / fail / skip counts
- For each failure: test ID + first 5 lines of traceback + minimal repro hint
- Wall-clock time

Do not dump full pytest output unless explicitly requested.
