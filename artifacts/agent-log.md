# Agent log — 2026-05-19 pre-flight 01 implementation session

## Deliverables produced

### D1 — H5 schemas (two)

- `src/brainrepa_fm/data/h5_schemas.py` — declarative dataclasses for both schemas.
- `src/brainrepa_fm/data/brats2026_schema.py` — Schema A (`brats_inpainting_2026.h5`) validator pair.
- `src/brainrepa_fm/data/brainrepa_latents_schema.py` — Schema B (`brainrepa_latents.h5`) validator pair (forward-declared).
- `src/brainrepa_fm/data/exceptions.py` — typed exceptions.
- `docs/h5-schemas.md` — human-readable design note.

### D2 — Source converter

- `src/brainrepa_fm/data/brats2026_converter.py` — `BraTS2026Converter`, atomic write + validate-on-close.
- `src/brainrepa_fm/data/brats_partition.py` — patient-level partitioner.
- `routines/data/brats2026_convert/{cli, configs/{default,smoke}, engine/converter_engine}.py`.
- Console script `brainrepa-data-brats2026-convert`.
- Verified on 12-scan smoke (8 train + 4 challenge-val) in ~5 s; validator passes.

### D3 — Augmentation preflight

- `src/brainrepa_fm/common/maisi.py` — `MaisiVAE` wrapper, fp16 autocast + activation checkpointing, two pad envelopes (`(256, 256, 192)` for A100, `(192, 192, 144)` for the local 3060), latent-shape probe.
- `src/brainrepa_fm/preflight/augmentation/transforms.py` — 8 transforms + simplified BraTS sampler.
- `src/brainrepa_fm/preflight/augmentation/vae_composability.py` — Δ_aug-VAE in 3 regions.
- `src/brainrepa_fm/preflight/augmentation/mask_audit.py` — 4 descriptors + KS test.
- `src/brainrepa_fm/preflight/augmentation/visualize.py` — 16 QC PNG grids + 4 KS CDFs.
- `routines/preflights/augmentation/{cli, configs/{default,smoke}, engine/augmentation_engine, slurm/{launcher,worker}_augmentation.sh}`.
- Console script `brainrepa-preflight-augmentation`.
- Verified on 4-train-volume smoke in ~6 min; produces decision.json + 16 + 4 figures + 3 CSVs + report.md.

### Tests

29 / 29 green:
- 27 unit (`tests/data/test_brats2026_schema.py`, `tests/data/test_brainrepa_latents_schema.py`, `tests/data/test_brats_partition.py`, `tests/preflight/augmentation/test_transforms.py`, `tests/preflight/augmentation/test_mask_audit.py`).
- 2 GPU/preflight (`tests/common/test_maisi.py`, `tests/data/test_brats2026_converter.py`).

## Deviations from the prompt (logged in DECISIONS.md)

1. **MAISI 3060 envelope.** 256³ does not fit on the 3060 (decode OOM). Falls back to `(192, 192, 144)` → latent `(4, 48, 48, 36)`. The A100 SLURM script keeps the full `(4, 64, 64, 48)` target.
2. **Simplified BraTS sampler.** A.2 and A.3 use a geometric approximation, not the verbatim official sampler. The smoke trips `ks_hard_fail` on volume because of this; the full-vendoring task is logged for a follow-up before Wk 3.
3. **decision.json key schema.** Uses the prompt/`preflight-pattern.md` keys (`include`, `drop`, `halve_range`, `vae_composition_gap_db`, `ks_p_values`, `ks_hard_fail`), not the spec §7 names. User-approved at clarification.
4. **Smoke wall-clock.** ~6 min on the 3060 vs the ≤ 5 min target — accepted as the floor with all deliverables.

## Outstanding / next steps for the downstream agent

1. **Full conversion + default preflight.** Run `brainrepa-data-brats2026-convert routines/data/brats2026_convert/configs/default.yaml` (~2-3 h on local disk) then `brainrepa-preflight-augmentation routines/preflights/augmentation/configs/default.yaml` (~6-8 h on the 3060) to produce a non-smoke `decision.json` over 50 stratified training subjects and the full 219 challenge-val.
2. **Vendor the official BraTS sampler.** Replace the simplified geometric sampler with the upstream `getHealthyMasks` at `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/2026_challenge/dataset/include.py` before Wk 3.
3. **Schema B producer.** The latent-H5 producer is forward-declared but unwritten. Trigger task: post-decision-json, the consumer reads the `include` list and encodes per-augmentation latents into the schema laid out in `docs/h5-schemas.md`.
4. **Pre-flight 02 (BSF layers)** and **pre-flight 03 (MAISI VAE audit)** — both consume `src/brainrepa_fm/common/maisi.py` as-is.
5. **Picasso submission.** SLURM scripts exist at `routines/preflights/augmentation/slurm/`. The Singularity image path is a placeholder (`$SINGULARITY_IMG`) — update to the current NGC PyTorch image on Picasso before submission.

## Resources used

- Subagents: 3 Explore agents at planning time (proposal/checks digest, repo state map, MAISI/sampler upstream code).
- No Plan / Read-paper / dl-scientist agents invoked — the prompt was prescriptive enough.
- No SLURM submission (out of scope per prompt §6).
