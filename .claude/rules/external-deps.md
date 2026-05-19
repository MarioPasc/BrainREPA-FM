# External Dependencies

This project depends on two **frozen** pretrained models (MAISI-V2 VAE-GAN, BrainSegFounder-Small) and the BraTS 2026 inpainting dataset. None of these live in the repo. Their canonical paths are listed in `src/external/README.md` and mirrored in the top-level `CLAUDE.md`.

## Hard rules

1. **Never edit code under `src/external/`** except `src/external/README.md` itself. The directory is a pointer index, not a vendored copy. The deny list in `.claude/settings.json` enforces this.
2. **Never write to checkpoint paths.** The deny list also blocks writes under `/media/mpascual/Sandisk2TB/checkpoints/**`. Treat the checkpoints as read-only system files.
3. **Adapter wrappers go in `src/brainrepa_fm/adapters/`**, not in any external source tree. Examples:
   - `src/brainrepa_fm/adapters/bsf_t1.py` — T1-only patch-embed rewrite around BrainSegFounder-Small (`in_channels: 2 → 1`, modes: `discard_t2` / `average`).
   - `src/brainrepa_fm/common/maisi.py` — shared MAISI VAE encode/decode primitive wrapper.
4. **External code is imported, not modified.** If an upstream change is needed (e.g. SwinUNETR `in_channels` swap), do it via subclass / monkey-patch / weight surgery in an adapter module — never by editing the external source.
5. **Dataset paths come from config.** Routines accept dataset paths as YAML parameters (default `/media/mpascual/MeningD2/INPAINTING/2026/...`). Never hard-code these paths in library code; the constant lives in the routine's `default.yaml`.
6. **Treat checkpoints as inputs to checksum.** When loading, log the file's SHA-256 (or its size + mtime if SHA is too slow) so the artifact's `decision.json` can record which exact weights produced the result.

## Canonical external paths (snapshot — `src/external/README.md` is source of truth)

| Asset | Path |
|---|---|
| BrainSegFounder source code | `/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/code/BrainSegFounder` |
| BrainSegFounder SSL checkpoint (UK Biobank, 43k subjects) | `/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt` |
| MAISI-V2 source code | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/code/NV-Generate-CTMR` |
| MAISI-V2 VAE-GAN checkpoint (encoder + decoder) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI-V2 Flow-Matching checkpoint (reference, not strictly required) | `/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt` |
| BraTS 2026 training data (NIfTI) | `/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training` |
| BraTS 2026 validation data (no GT) | `/media/mpascual/MeningD2/INPAINTING/2026/source/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation` |
| BraTS 2026 H5 cache | `/media/mpascual/MeningD2/INPAINTING/2026/h5/brats_inpainting_2026.h5` |
| Documentation (proposal + checks) | `/media/mpascual/Sandisk2TB/research/brainrepa_fm/docs/` |

When this table drifts from `src/external/README.md`, **`src/external/README.md` wins** and this file should be updated. CI may add a check that the two are in sync.

## Adapter module checklist

When you create a new adapter:

- [ ] Lives under `src/brainrepa_fm/adapters/` (or `src/brainrepa_fm/common/` for cross-cutting primitives).
- [ ] Imports from the external source via the path declared in `src/external/README.md` (or via an installed pip package if upstream publishes one — preferred when available).
- [ ] Carries a module-level docstring stating which external commit / checkpoint version it targets.
- [ ] Has a unit test in `tests/adapters/test_<name>.py` that exercises the wrapper with a synthetic input (no checkpoint download).
- [ ] Logs the resolved checkpoint path and its checksum at first load.
