"""Pre-flight 03 — MAISI VAE-GAN reconstruction audit (library code).

Implements the reconstruction-fidelity portion of
``docs/checks/03_maisi_vae_audit.md``: round-trip MSE/PSNR/SSIM, per-channel
latent statistics (Caveat 8), and the §7 voided-encoder behaviour tests. The
equivariance audit (§2.2, §3.4-3.7) is deferred — see ``DECISIONS.md``.

Every public function here is pure (no VAE, no GPU): the routine engine owns
the encode/decode and feeds decoded volumes / latents in. ``visualize`` is
imported directly by the engine, not re-exported here, to keep this package's
import light.
"""

from brainrepa_fm.preflight.maisi_vae.latent_stats import (
    LatentChannelStats,
    LatentStatsAccumulator,
)
from brainrepa_fm.preflight.maisi_vae.preprocess import prepare_to_envelope
from brainrepa_fm.preflight.maisi_vae.reconstruction import (
    ReconstructionMetrics,
    VoidedRoundtripMetrics,
    compute_reconstruction_metrics,
    compute_voided_roundtrip_metrics,
)
from brainrepa_fm.preflight.maisi_vae.voided_tests import (
    VoidedTestResult,
    compute_voided_tests_from_latents,
    downsample_mask_to_latent,
)

__all__ = [
    "LatentChannelStats",
    "LatentStatsAccumulator",
    "ReconstructionMetrics",
    "VoidedRoundtripMetrics",
    "VoidedTestResult",
    "compute_reconstruction_metrics",
    "compute_voided_roundtrip_metrics",
    "compute_voided_tests_from_latents",
    "downsample_mask_to_latent",
    "prepare_to_envelope",
]
