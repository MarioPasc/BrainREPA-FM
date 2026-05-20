# Routines & Pre-Flight Pattern

A *routine* is a runnable task (run a pre-flight check, train a baseline, evaluate a model, …) that wraps a configurable engine. All routines live under `routines/` and follow the same layout.

```
routines/
├── preflights/                 # pre-flight gates (must pass before Stage-2 FM training)
│   ├── augmentation/
│   ├── bsf_layers/
│   └── maisi_vae/
└── <stage>/                    # Wk 1-12 training/eval routines (see training-stages.md)
```

Each routine has the same internal layout:

```
routines/<bucket>/<name>/
├── __init__.py
├── cli.py                      # entrypoint: `python -m routines.<bucket>.<name>.cli <yaml>`
├── configs/                    # one YAML per concrete invocation
│   ├── default.yaml
│   └── smoke.yaml              # optional, fast local sanity-check
├── slurm/                      # Picasso submission scripts (Singularity, no Docker)
│   ├── launcher_<name>.sh
│   └── worker_<name>.sh
└── engine/
    ├── __init__.py             # re-exports `<Name>Engine` and `<Name>RoutineConfig`
    └── <name>_engine.py        # thin orchestrator that imports library code from src/
```

**Library implementations live in `src/brainrepa_fm/<area>/<name>/`** (e.g. `src/brainrepa_fm/preflight/augmentation/{transforms,vae_composability,mask_audit,visualize}.py`). The engine module is a **thin wrapper** that wires library functions to the YAML config, runs them in order, and writes the artifact. This separation keeps library code importable and unit-testable without invoking the CLI.

## Invariants

1. **`cli.py` takes one positional argument**: the path to a YAML config. No other flags. Logging level is read from the YAML.
2. **The engine module exports two public symbols**: a frozen `<Name>RoutineConfig` dataclass (with a `from_yaml(path)` classmethod) and an `<Name>Engine` class with a single `run() -> Path` method that returns the produced artifact path.
3. **Configs are reproducible.** Persist every parameter that influenced the output into the artifact directory: a copy of the resolved YAML, an ISO-8601 timestamp, the git commit SHA, the resolved checkpoint paths, the env name (`brainrepa`).
4. **Validate on close.** If the routine produces an H5, the engine calls `assert_<artifact>_valid(path)` (see `h5-design-principles.md`) before returning. If it produces another artifact type (JSON, figures), assert that every deliverable specified in the spec under `docs/checks/` is present.
5. **Console scripts.** Register each routine in `pyproject.toml` `[project.scripts]` as `brainrepa-<bucket>-<name>` (e.g. `brainrepa-preflight-maisi-vae = "routines.preflights.maisi_vae.cli:main"`) so both forms work:
   - `brainrepa-preflight-maisi-vae cfg.yaml`
   - `python -m routines.preflights.maisi_vae.cli cfg.yaml`
6. **No heavy work at import time.** `cli.py` and `engine/__init__.py` must not load checkpoints, instantiate models, or call `cuda` at module scope. All side effects live inside `Engine.run()`.
7. **One routine, one responsibility.** Split modes into separate routines that share a library module — never add a multi-mode flag.

## Pre-flight specifics

Pre-flight checks are defined in `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/checks/` (specs `00_README.md` through `03_maisi_vae_audit.md`). **The docs are the source of truth**; this rule pins the contract between pre-flights and downstream routines.

Dependency order: **03 → (01 ∥ 02) → Stage-2 FM training**.

Each pre-flight routine writes its deliverables to:

```
artifacts/<routine>/<UTC-timestamp>/
├── report.md           # human-readable, includes inlined figures
├── figures/            # PNGs / PDFs referenced by report.md
├── tables/             # CSVs of raw numbers
└── decision.json       # MACHINE-READABLE contract for downstream consumers
```

`decision.json` is the **machine-readable contract** consumed by downstream training routines. Schema per check (sourced from `docs/checks/`):

### `preflights/maisi_vae` (proposal §03)
```json
{
  "schema_version": "1.0",
  "path": "1|2|3",
  "vae_fine_tune": true,
  "fine_tune_target": "brain|none",
  "latent_aug_safe": ["bias_field", "gamma", "intensity_shift"],
  "latent_scale": [s0, s1, s2, s3],
  "median_inside_void_psnr_db": 28.4,
  "tumor_vs_brain_gap_db": 1.2
}
```
Drives the architecture-path decision: Path 1 frozen VAE / Path 1 + fine-tune / Path 3 wavelet (fastWDM3D / FlowLet).

### `preflights/bsf_layers` (proposal §02)
```json
{
  "schema_version": "2.0",
  "feasible": true,
  "ell_star": 1,
  "ell_runner_up": 2,
  "candidate_stages": [0, 1, 2, 3, 4],
  "stage_shape_at_ell_star": {"channels": 96, "grid": [24, 24, 24]},
  "cknna_y2_at_ell_star": 0.13,
  "cknna_y2_null_p95": 0.04,
  "token_probe_p1_r2": 0.71,
  "ssl_recon_psnr_db": 31.2,
  "resample_mode": "trilinear",
  "checkpoint_path": "/abs/path/to/64-gpu-model_bestValRMSE.pt",
  "checkpoint_sha256": "<sha256>",
  "checkpoint_in_channels": 1,
  "checkpoint_global_step": 75
}
```
Selects the BSF REPA target stage. `ell_star` indexes the five `swinViT.forward()`
outputs, `ell ∈ {0,…,4}` (0 = patch-embed, 4 = bottleneck). No T1 adaptation key:
the released BSF-S checkpoint is natively single-channel (`in_channels = 1`).
`feasible == false` is a hard fail — BSF is not a usable REPA encoder and the
project routes to a fallback. Schema bumped to `2.0`: the `1.0` `adaptation_mode`
key is dropped and `ell_star` is re-indexed `0-4` (was `1-4`).

### `preflights/augmentation` (proposal §01)
```json
{
  "schema_version": "1.0",
  "include": ["A.1", "A.2", "A.3", "B.1", "C.1", "C.2", "C.3"],
  "drop": ["C.4"],
  "halve_range": ["C.1"],
  "vae_composition_gap_db": {"C.1": 0.42, "C.2": 0.18, ...},
  "ks_p_values": {"volume": 0.31, "sv_ratio": 0.12, "centroid": 0.08, "max_diameter": 0.21},
  "ks_hard_fail": false
}
```
Drives the final augmentation set used by `fm_baseline` / `fm_repa` / `fm_repa_pixel`.

A downstream consumer never reads `report.md` programmatically. It loads `decision.json`, asserts `schema_version`, and uses the keys. `report.md` exists for the human reviewer.

## Hard rules

- **Pre-flights are gating.** A training routine for Wk 3+ must, at startup, load each pre-flight's `decision.json` and assert the conditions it depends on (e.g. `path == "1"` for the frozen-VAE branch, `feasible == true` and `ell_star in {1,2,3}` for the BSF REPA target). If a pre-flight has not run, the routine fails fast with a clear message naming the missing artifact.
- **Pre-flight outputs are immutable once written.** A re-run produces a new timestamped directory under `artifacts/<routine>/`. Never overwrite.
- **Latest pointer.** `artifacts/<routine>/LATEST` is a symlink to the most-recent timestamped directory. Consumers default to following the symlink and can be pinned to a specific timestamp via the YAML config (`preflight_artifact_path: artifacts/preflights/maisi_vae/2026-05-20T14-32-00Z/`).

## Reference

Once `routines/preflights/augmentation/` lands, it becomes the canonical example of this pattern. New routines copy its layout.
