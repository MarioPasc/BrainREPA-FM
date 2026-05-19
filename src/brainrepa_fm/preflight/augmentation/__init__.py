"""Pre-flight 01 — augmentation composability with the MAISI VAE.

Library modules consumed by ``routines.preflights.augmentation``:

- :mod:`brainrepa_fm.preflight.augmentation.transforms` — the eight transforms
  (A.1, A.2, A.3, B.1, C.1, C.2, C.3, C.4) as :class:`TransformSpec` instances
  plus a simplified void-mask sampler.
- :mod:`brainrepa_fm.preflight.augmentation.vae_composability` — computes
  ``Δ_aug-VAE`` per (transform, region).
- :mod:`brainrepa_fm.preflight.augmentation.mask_audit` — four mask descriptors
  (volume, surface-to-volume, centroid distance, max diameter) and KS test.
- :mod:`brainrepa_fm.preflight.augmentation.visualize` — the 16 QC PNG grids
  and four KS CDF figures.
"""
