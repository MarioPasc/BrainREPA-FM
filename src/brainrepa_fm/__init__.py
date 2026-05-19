"""BrainREPA-FM — BraTS-Lighthouse 2026 inpainting.

3D latent flow-matching generator in the MAISI-V2 VAE latent space, aligned to
frozen BrainSegFounder-Small features via REPA, with a pixel-space L1+(1-SSIM)
head on the decoded volume.

See ``CLAUDE.md`` (repo root) for paths, conventions, and rule pointers.
"""

__version__ = "0.0.1"
