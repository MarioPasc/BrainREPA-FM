"""Pre-flight 03 engine — MAISI VAE-GAN reconstruction audit.

Reads the source ``brats_inpainting_2026.h5`` (Schema A), selects a subset of
training subjects that carry ground truth, round-trips each intact volume
through the frozen MAISI-V2 VAE, and writes the artifact bundle:

``artifacts/preflights/maisi_vae/<UTC>/``
    ├── report.md
    ├── figures/{psnr_hist_*,ssim_hist_*,montage_*,latent_statistics,
    │            voided_tests_scatter,voided_roundtrip_psnr_drop}.png
    ├── tables/{reconstruction_metrics,latent_stats,voided_tests,voided_roundtrip}.csv
    └── decision.json

Scope follows ``docs/checks/03_maisi_vae_audit.md`` §2.1 / §3 / §4 (reconstruction)
plus §7 (voided-encoder behaviour) and Caveat 8 (latent statistics). The
equivariance audit (§2.2, §3.4-3.7) is deferred — see ``DECISIONS.md``.

The engine owns every encode/decode; all metric maths live in the pure library
modules under ``src/brainrepa_fm/preflight/maisi_vae/``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from brainrepa_fm.common.maisi import (
    MAISI_PAD_SHAPE,
    MAISI_PAD_SHAPE_3060,
    MaisiVAE,
    probe_latent_shape,
    tensor_to_volume,
    volume_to_tensor,
)
from brainrepa_fm.data.exceptions import MaisiAuditError, PreflightError
from brainrepa_fm.preflight.augmentation.transforms import sample_void_mask
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
from brainrepa_fm.preflight.maisi_vae.visualize import (
    render_latent_stats_figure,
    render_psnr_histogram,
    render_reconstruction_montage,
    render_ssim_histogram,
    render_voided_drop_histogram,
    render_voided_roundtrip_montage,
    render_voided_scatter,
)
from brainrepa_fm.preflight.maisi_vae.voided_tests import (
    VoidedTestResult,
    compute_voided_tests_from_latents,
    downsample_mask_to_latent,
)

logger = logging.getLogger(__name__)

# Producer identifier persisted into decision.json.
PRODUCER_ID: str = "routines.preflights.maisi_vae:v0.0.1"
DECISION_SCHEMA_VERSION: str = "1.0"

# Decision-rule thresholds (docs/checks/03_maisi_vae_audit.md §3.3 / §4).
PSNR_PATH1_DB: float = 28.0
PSNR_CONTINGENCY_DB: float = 24.0
TUMOR_GAP_FLAG_DB: float = 5.0


# ---------------------------------------------------------------------------
# Routine config
# ---------------------------------------------------------------------------


class MaisiVaeRoutineConfig(BaseModel):
    """Pydantic config for the MAISI VAE reconstruction audit (pre-flight 03)."""

    source_h5: Path
    output_root: Path = Path("artifacts/preflights/maisi_vae")
    n_subjects: int | None = None  # None → all training volumes with GT
    target_shape: Literal["3060", "a100"] = "3060"
    n_void_masks_per_volume: int = Field(default=10, ge=1)  # J in docs/checks/03 §2.1
    void_seed: int = 2026
    seed: int = 2026
    device: str = "cuda"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Optional MAISI overrides — when unset, MaisiVAE falls back to its module-level
    # defaults (local-workstation paths). Set explicitly on Picasso.
    maisi_checkpoint_path: Path | None = None
    maisi_config_path: Path | None = None

    @field_validator("source_h5")
    @classmethod
    def _resolve_source(cls, v: Path) -> Path:
        p = Path(v).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"source H5 not found: {p}")
        return p

    @field_validator("output_root")
    @classmethod
    def _resolve_output(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("n_subjects")
    @classmethod
    def _check_n_subjects(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"n_subjects must be >= 1 or null, got {v}")
        return v


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MaisiVaeEngine:
    """YAML-driven orchestrator for pre-flight 03."""

    def __init__(self, config: MaisiVaeRoutineConfig) -> None:
        self.config = config
        logger.setLevel(config.log_level)

    # -- main entry point -------------------------------------------------

    def run(self) -> Path:
        cfg = self.config
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = cfg.output_root / ts
        figures_dir = out_dir / "figures"
        tables_dir = out_dir / "tables"
        for d in (out_dir, figures_dir, tables_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("output directory: %s", out_dir)

        git_sha = _git_sha()
        target_shape = MAISI_PAD_SHAPE_3060 if cfg.target_shape == "3060" else MAISI_PAD_SHAPE

        with h5py.File(cfg.source_h5, "r") as f:
            cohort = self._select_cohort(f, cfg)
            scan_ids = np.array(
                [s.decode() if isinstance(s, bytes) else s for s in f["scan_id"][...]]
            )
            gt_scan_index = f["gt/scan_index"][...].astype(int)
            global_to_gt = {int(g): i for i, g in enumerate(gt_scan_index)}
            logger.info("audit cohort: %d training volumes with GT", len(cohort))

            # ---- step 1: instantiate VAE & probe the latent grid ----------
            logger.info(
                "loading MAISI VAE (target_shape=%s, ckpt=%s)",
                target_shape,
                cfg.maisi_checkpoint_path,
            )
            vae = MaisiVAE(
                checkpoint_path=cfg.maisi_checkpoint_path,
                config_path=cfg.maisi_config_path,
                device=cfg.device,
                autocast_fp16=True,
                use_checkpointing=True,
            )
            latent_shape = probe_latent_shape(vae, input_shape=target_shape, device=cfg.device)
            latent_channels = latent_shape[0]
            latent_grid = latent_shape[1:]
            logger.info("VAE sha256[:16]=%s, latent shape=%s", vae.info.sha256_prefix, latent_shape)

            # ---- step 2: per-volume single-encode audit -------------------
            recon_metrics: list[ReconstructionMetrics] = []
            voided_results: list[VoidedTestResult] = []
            voided_rt_metrics: list[VoidedRoundtripMetrics] = []
            latent_acc = LatentStatsAccumulator(latent_channels)

            for pos, scan_idx in enumerate(cohort):
                sid = str(scan_ids[scan_idx])
                gt_local = global_to_gt[scan_idx]
                gt = prepare_to_envelope(f["gt/t1"][gt_local], target_shape)
                voided = prepare_to_envelope(f["images/t1_voided"][scan_idx], target_shape)
                brain = prepare_to_envelope(f["masks/brain"][scan_idx], target_shape)
                tumor = prepare_to_envelope(f["gt/tumor_mask"][gt_local], target_shape)
                # The H5's own void mask (the one that produced images/t1_voided),
                # distinct from the J freshly-sampled void_masks below.
                hvoid = prepare_to_envelope(f["masks/void"][scan_idx], target_shape)

                void_masks = [
                    sample_void_mask(
                        brain,
                        tumor=None,
                        widen_factor=1.0,
                        seed=cfg.void_seed + pos * cfg.n_void_masks_per_volume + j,
                    )
                    for j in range(cfg.n_void_masks_per_volume)
                ]

                # Encode the intact volume exactly once; reuse for everything.
                z_gt = vae.encode(volume_to_tensor(gt, device=cfg.device))
                recon = tensor_to_volume(vae.decode(z_gt).float())
                recon_metrics.append(
                    compute_reconstruction_metrics(
                        subject_id=sid,
                        gt=gt,
                        recon=recon,
                        brain_mask=brain,
                        tumor_mask=tumor,
                        void_masks=void_masks,
                    )
                )
                latent_acc.update(z_gt)

                # §7 voided-encoder behaviour + Caveat 2 voided round-trip —
                # one extra encode + decode of the voided volume.
                z_void = vae.encode(volume_to_tensor(voided, device=cfg.device))
                recon_voided = tensor_to_volume(vae.decode(z_void).float())
                latent_void_masks = [
                    downsample_mask_to_latent(vm, latent_grid) for vm in void_masks
                ]
                voided_results.append(
                    compute_voided_tests_from_latents(
                        subject_id=sid,
                        z_gt=z_gt,
                        z_voided=z_void,
                        latent_void_masks=latent_void_masks,
                    )
                )
                voided_rt_metrics.append(
                    compute_voided_roundtrip_metrics(
                        subject_id=sid,
                        gt=gt,
                        voided=voided,
                        recon_unvoided=recon,
                        recon_voided=recon_voided,
                        brain_mask=brain,
                        void_mask=hvoid,
                    )
                )

                del z_gt, z_void
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (pos + 1) % 10 == 0 or pos + 1 == len(cohort):
                    logger.info("audited %d/%d volumes", pos + 1, len(cohort))

            if not recon_metrics:
                raise MaisiAuditError("audit produced no metrics — empty cohort")

            latent_stats = latent_acc.result()

            # ---- step 3: decision rule ------------------------------------
            decision = self._decide(
                recon_metrics,
                voided_rt_metrics,
                latent_stats,
                vae=vae,
                cfg=cfg,
                git_sha=git_sha,
                ts=ts,
            )

            # ---- step 4: write artifact bundle ----------------------------
            montages = self._collect_montage_volumes(
                vae, f, recon_metrics, cohort, global_to_gt, target_shape, cfg
            )
            voided_montages = self._collect_voided_montages(
                vae, f, voided_rt_metrics, cohort, global_to_gt, target_shape, cfg
            )

        self._write_tables(
            tables_dir, recon_metrics, voided_results, voided_rt_metrics, latent_stats
        )
        self._write_figures(
            figures_dir,
            recon_metrics,
            voided_results,
            voided_rt_metrics,
            latent_stats,
            montages,
            voided_montages,
        )
        (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
        self._write_report(
            out_dir, recon_metrics, voided_results, voided_rt_metrics, latent_stats, decision
        )
        self._update_latest_symlink(cfg.output_root, ts)

        # ---- step 5: validate-on-close --------------------------------
        self._assert_deliverables(out_dir, tables_dir)

        # ---- step 6: hard-fail gate -----------------------------------
        if decision["path"] == "3":
            msg = (
                f"MAISI VAE reconstruction hard-fail: median inside-void PSNR "
                f"{decision['median_inside_void_psnr_db']} dB < {PSNR_CONTINGENCY_DB} dB — "
                f"route to Path 3 (wavelet-domain flow matching)"
            )
            logger.error("%s", msg)
            raise PreflightError(msg)

        logger.info(
            "decision.json written: %s (path=%s)", out_dir / "decision.json", decision["path"]
        )
        return out_dir

    # -- helpers ----------------------------------------------------------

    def _select_cohort(self, f: h5py.File, cfg: MaisiVaeRoutineConfig) -> list[int]:
        """Training volumes that carry ground truth, sub-sampled to ``n_subjects``."""
        train_idx = (
            f["splits/train"][...].astype(int) if "splits/train" in f else np.array([], dtype=int)
        )
        gt_idx = (
            f["gt/scan_index"][...].astype(int) if "gt/scan_index" in f else np.array([], dtype=int)
        )
        cohort_all = sorted(set(int(i) for i in train_idx) & set(int(i) for i in gt_idx))
        if not cohort_all:
            raise MaisiAuditError(
                "no training volumes with ground truth: splits/train ∩ gt/scan_index is empty"
            )
        n = len(cohort_all) if cfg.n_subjects is None else min(cfg.n_subjects, len(cohort_all))
        rng = np.random.default_rng(cfg.seed)
        picks = rng.choice(np.array(cohort_all), size=n, replace=False)
        return sorted(int(i) for i in picks)

    def _decide(
        self,
        recon_metrics: list[ReconstructionMetrics],
        voided_rt_metrics: list[VoidedRoundtripMetrics],
        latent_stats: LatentChannelStats,
        *,
        vae: MaisiVAE,
        cfg: MaisiVaeRoutineConfig,
        git_sha: str,
        ts: str,
    ) -> dict[str, object]:
        """Apply the §3.3 / §4 reconstruction decision rule."""

        def median(attr: str) -> float:
            vals = np.array([getattr(m, attr) for m in recon_metrics], dtype=np.float64)
            finite = vals[~np.isnan(vals)]
            return float(np.median(finite)) if finite.size else float("nan")

        median_void = median("psnr_void_mean")
        median_brain = median("psnr_brain")
        median_full = median("psnr_full")
        median_tumor = median("psnr_tumor")
        tumor_gap = median_brain - median_tumor

        drop_vals = np.array([m.delta_psnr_visible_db for m in voided_rt_metrics], dtype=np.float64)
        drop_finite = drop_vals[~np.isnan(drop_vals)]
        median_voided_drop = float(np.median(drop_finite)) if drop_finite.size else float("nan")

        if median_void >= PSNR_PATH1_DB:
            path, fine_tune, ft_target = "1", False, "none"
        elif median_void >= PSNR_CONTINGENCY_DB:
            path, fine_tune, ft_target = "1", True, "brain"
        else:
            path, fine_tune, ft_target = "3", True, "brain"
        investigate = bool(np.isfinite(tumor_gap) and tumor_gap > TUMOR_GAP_FLAG_DB)

        return {
            "schema_version": DECISION_SCHEMA_VERSION,
            "producer": PRODUCER_ID,
            "git_sha": git_sha,
            "timestamp_utc": ts,
            "path": path,
            "vae_fine_tune": fine_tune,
            "fine_tune_target": ft_target,
            # latent_aug_safe is derived from the equivariance audit, which is
            # deferred — the empty list signals "augment in image space only".
            "latent_aug_safe": [],
            "latent_scale": [_json_num(s, 6) for s in latent_stats.std],
            "latent_mean": [_json_num(m, 6) for m in latent_stats.mean],
            "median_inside_void_psnr_db": _json_num(median_void),
            "median_brain_psnr_db": _json_num(median_brain),
            "median_full_psnr_db": _json_num(median_full),
            "median_tumor_psnr_db": _json_num(median_tumor),
            "tumor_vs_brain_gap_db": _json_num(tumor_gap),
            "investigate_tumor_gap": investigate,
            "median_voided_visible_psnr_drop_db": _json_num(median_voided_drop),
            "n_volumes_audited": len(recon_metrics),
            "target_shape": cfg.target_shape,
            "vae_sha256_prefix": vae.info.sha256_prefix,
        }

    def _collect_montage_volumes(
        self,
        vae: MaisiVAE,
        f: h5py.File,
        recon_metrics: list[ReconstructionMetrics],
        cohort: list[int],
        global_to_gt: dict[int, int],
        target_shape: tuple[int, int, int],
        cfg: MaisiVaeRoutineConfig,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Re-encode the best / median / worst volume (by inside-void PSNR) for montages."""
        psnrs = np.array([m.psnr_void_mean for m in recon_metrics], dtype=np.float64)
        finite = np.where(~np.isnan(psnrs))[0]
        if finite.size == 0:
            finite = np.arange(len(recon_metrics))
        order = finite[np.argsort(psnrs[finite])]
        chosen = {
            "worst": int(order[0]),
            "median": int(order[order.size // 2]),
            "best": int(order[-1]),
        }
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label, k in chosen.items():
            scan_idx = cohort[k]
            gt_local = global_to_gt[scan_idx]
            gt = prepare_to_envelope(f["gt/t1"][gt_local], target_shape)
            brain = prepare_to_envelope(f["masks/brain"][scan_idx], target_shape)
            z = vae.encode(volume_to_tensor(gt, device=cfg.device))
            recon = tensor_to_volume(vae.decode(z).float())
            void_mask = sample_void_mask(
                brain,
                tumor=None,
                widen_factor=1.0,
                seed=cfg.void_seed + k * cfg.n_void_masks_per_volume,
            )
            out[label] = (gt, recon, void_mask)
            del z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

    def _collect_voided_montages(
        self,
        vae: MaisiVAE,
        f: h5py.File,
        voided_rt_metrics: list[VoidedRoundtripMetrics],
        cohort: list[int],
        global_to_gt: dict[int, int],
        target_shape: tuple[int, int, int],
        cfg: MaisiVaeRoutineConfig,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Re-encode 3 volumes (least / median / worst visible-PSNR drop) for voided montages."""
        drops = np.array([m.delta_psnr_visible_db for m in voided_rt_metrics], dtype=np.float64)
        finite = np.where(~np.isnan(drops))[0]
        if finite.size == 0:
            finite = np.arange(len(voided_rt_metrics))
        order = finite[np.argsort(drops[finite])]  # ascending visible-PSNR drop
        chosen = {
            "least_drop": int(order[0]),
            "median_drop": int(order[order.size // 2]),
            "worst_drop": int(order[-1]),
        }
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for label, k in chosen.items():
            scan_idx = cohort[k]
            gt_local = global_to_gt[scan_idx]
            gt = prepare_to_envelope(f["gt/t1"][gt_local], target_shape)
            voided = prepare_to_envelope(f["images/t1_voided"][scan_idx], target_shape)
            hvoid = prepare_to_envelope(f["masks/void"][scan_idx], target_shape)
            z = vae.encode(volume_to_tensor(voided, device=cfg.device))
            recon_voided = tensor_to_volume(vae.decode(z).float())
            out[label] = (gt, voided, recon_voided, hvoid)
            del z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

    def _write_tables(
        self,
        tables_dir: Path,
        recon_metrics: list[ReconstructionMetrics],
        voided_results: list[VoidedTestResult],
        voided_rt_metrics: list[VoidedRoundtripMetrics],
        latent_stats: LatentChannelStats,
    ) -> None:
        recon_cols = [fld.name for fld in fields(ReconstructionMetrics)]
        with (tables_dir / "reconstruction_metrics.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(recon_cols)
            for m in recon_metrics:
                d = asdict(m)
                writer.writerow([_csv_cell(d[c]) for c in recon_cols])

        void_cols = [fld.name for fld in fields(VoidedTestResult)]
        with (tables_dir / "voided_tests.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(void_cols)
            for r in voided_results:
                d = asdict(r)
                writer.writerow([_csv_cell(d[c]) for c in void_cols])

        vrt_cols = [fld.name for fld in fields(VoidedRoundtripMetrics)]
        with (tables_dir / "voided_roundtrip.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(vrt_cols)
            for m in voided_rt_metrics:
                d = asdict(m)
                writer.writerow([_csv_cell(d[c]) for c in vrt_cols])

        with (tables_dir / "latent_stats.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["channel", "mean", "std"])
            for c, (mu, sd) in enumerate(zip(latent_stats.mean, latent_stats.std, strict=True)):
                writer.writerow([c, _csv_cell(mu), _csv_cell(sd)])

    def _write_figures(
        self,
        figures_dir: Path,
        recon_metrics: list[ReconstructionMetrics],
        voided_results: list[VoidedTestResult],
        voided_rt_metrics: list[VoidedRoundtripMetrics],
        latent_stats: LatentChannelStats,
        montages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        voided_montages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        psnr_panels = (
            ("psnr_hist_full.png", "full volume", "psnr_full", None),
            ("psnr_hist_brain.png", "brain", "psnr_brain", None),
            ("psnr_hist_void.png", "inside void", "psnr_void_mean", PSNR_PATH1_DB),
            ("psnr_hist_tumor.png", "tumor", "psnr_tumor", None),
        )
        for fname, label, attr, threshold in psnr_panels:
            render_psnr_histogram(
                [getattr(m, attr) for m in recon_metrics],
                region_label=label,
                threshold_db=threshold,
                out_path=figures_dir / fname,
            )

        ssim_panels = (
            ("ssim_hist_full.png", "full volume", "ssim_full"),
            ("ssim_hist_brain.png", "brain", "ssim_brain"),
            ("ssim_hist_void.png", "inside void", "ssim_void_mean"),
        )
        for fname, label, attr in ssim_panels:
            render_ssim_histogram(
                [getattr(m, attr) for m in recon_metrics],
                region_label=label,
                out_path=figures_dir / fname,
            )

        for label, (gt, recon, void_mask) in montages.items():
            render_reconstruction_montage(
                gt_volume=gt,
                reconstructed=recon,
                void_mask=void_mask,
                label=label,
                out_path=figures_dir / f"montage_{label}.png",
            )

        for label, (gt, voided, recon_voided, void_mask) in voided_montages.items():
            render_voided_roundtrip_montage(
                gt_volume=gt,
                voided_volume=voided,
                recon_voided=recon_voided,
                void_mask=void_mask,
                label=label,
                out_path=figures_dir / f"montage_voided_{label}.png",
            )

        render_latent_stats_figure(
            latent_stats.mean, latent_stats.std, out_path=figures_dir / "latent_statistics.png"
        )
        render_voided_scatter(
            [r.s_inside_mean for r in voided_results],
            [r.s_outside_mean for r in voided_results],
            out_path=figures_dir / "voided_tests_scatter.png",
        )
        render_voided_drop_histogram(
            [m.delta_psnr_visible_db for m in voided_rt_metrics],
            out_path=figures_dir / "voided_roundtrip_psnr_drop.png",
        )

    def _write_report(
        self,
        out_dir: Path,
        recon_metrics: list[ReconstructionMetrics],
        voided_results: list[VoidedTestResult],
        voided_rt_metrics: list[VoidedRoundtripMetrics],
        latent_stats: LatentChannelStats,
        decision: dict[str, object],
    ) -> None:
        lines = [
            "# Pre-flight 03 — MAISI VAE-GAN reconstruction audit",
            "",
            f"_Generated at {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
            f"by {PRODUCER_ID}._",
            "",
            f"- Volumes audited: {len(recon_metrics)}",
            f"- VAE envelope: {decision['target_shape']}",
            f"- VAE checkpoint sha256[:16]: {decision['vae_sha256_prefix']}",
            f"- git SHA: {decision['git_sha']}",
            "",
            "## Round-trip PSNR / SSIM (per-volume statistics)",
            "",
            "| Region | PSNR median | PSNR mean | PSNR p10 | PSNR p90 | SSIM median |",
            "|---|---|---|---|---|---|",
        ]
        regions = (
            ("full", "psnr_full", "ssim_full"),
            ("brain", "psnr_brain", "ssim_brain"),
            ("inside void", "psnr_void_mean", "ssim_void_mean"),
            ("tumor", "psnr_tumor", "ssim_tumor"),
        )
        for label, psnr_attr, ssim_attr in regions:
            ps = _summary([getattr(m, psnr_attr) for m in recon_metrics])
            ss = _summary([getattr(m, ssim_attr) for m in recon_metrics])
            lines.append(
                f"| {label} | {ps['median']:.2f} | {ps['mean']:.2f} | "
                f"{ps['p10']:.2f} | {ps['p90']:.2f} | {ss['median']:.4f} |"
            )

        s_in = _summary([r.s_inside_mean for r in voided_results])
        s_out = _summary([r.s_outside_mean for r in voided_results])
        vrt_drop = _summary([m.delta_psnr_visible_db for m in voided_rt_metrics])
        vrt_voided = _summary([m.psnr_visible_voided for m in voided_rt_metrics])
        vrt_unvoided = _summary([m.psnr_visible_unvoided for m in voided_rt_metrics])
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"- **path**: {decision['path']} (1 = frozen VAE, 2 = VAE fine-tune, 3 = wavelet)",
                f"- **vae_fine_tune**: {decision['vae_fine_tune']} "
                f"(target: {decision['fine_tune_target']})",
                f"- **median inside-void PSNR**: {decision['median_inside_void_psnr_db']} dB "
                f"(thresholds: ≥{PSNR_PATH1_DB} → Path 1; "
                f"{PSNR_CONTINGENCY_DB}-{PSNR_PATH1_DB} → Path 1 + fine-tune contingency; "
                f"<{PSNR_CONTINGENCY_DB} → Path 3)",
                f"- **tumor vs brain PSNR gap**: {decision['tumor_vs_brain_gap_db']} dB "
                f"(investigate if > {TUMOR_GAP_FLAG_DB}: {decision['investigate_tumor_gap']})",
                f"- **voided-input visible-region PSNR drop**: "
                f"{decision['median_voided_visible_psnr_drop_db']} dB (median over volumes)",
                f"- **latent_aug_safe**: {decision['latent_aug_safe']} "
                "(empty — equivariance audit deferred)",
                "",
                "## Latent statistics (Caveat 8)",
                "",
                "| Channel | mean | std |",
                "|---|---|---|",
            ]
        )
        for c, (mu, sd) in enumerate(zip(latent_stats.mean, latent_stats.std, strict=True)):
            lines.append(f"| {c} | {mu:.4f} | {sd:.4f} |")
        lines.extend(
            [
                "",
                "decision.json `latent_scale` = per-channel std, `latent_mean` = per-channel "
                "mean; both populate the Schema B placeholders.",
                "",
                "## §7 voided-encoder behaviour",
                "",
                f"- S_inside  (want large): median {s_in['median']:.4e}, "
                f"p10 {s_in['p10']:.4e}, p90 {s_in['p90']:.4e}",
                f"- S_outside (want ≈ 0):   median {s_out['median']:.4e}, "
                f"p10 {s_out['p10']:.4e}, p90 {s_out['p90']:.4e}",
                "",
                "## Voided-input round-trip (Caveat 2)",
                "",
                "Visible-region (brain minus void) round-trip fidelity when the VAE "
                "encodes the voided volume vs the intact volume:",
                "",
                f"- visible PSNR, un-voided input: median {vrt_unvoided['median']:.2f} dB",
                f"- visible PSNR, voided input:    median {vrt_voided['median']:.2f} dB",
                f"- PSNR drop (un-voided minus voided): median {vrt_drop['median']:.3f} dB "
                f"(p10 {vrt_drop['p10']:.3f}, p90 {vrt_drop['p90']:.3f})",
                "",
                "## Figures",
                "",
            ]
        )
        for fp in sorted((out_dir / "figures").glob("*.png")):
            lines.append(f"![{fp.name}](figures/{fp.name})")
        lines.extend(
            [
                "",
                "## Caveats",
                "",
                "- Round-trip measured on intact `gt/t1` (docs/checks/03 Caveat 1): this is "
                "the architectural ceiling on inside-void leaderboard metrics.",
                "- H5 volumes are pre-normalized to [0, 1] by the converter; no renormalization "
                "is applied here (deviation from §3.1 — see DECISIONS.md).",
                "- The voided-input round-trip (Caveat 2) scores D(E(x_tilde)) on the "
                "visible region; the architecture-path decision still uses only the "
                "intact round-trip.",
                "- Equivariance audit (§2.2, §3.4-3.7) is out of scope for this routine version.",
            ]
        )
        (out_dir / "report.md").write_text("\n".join(lines))

    def _assert_deliverables(self, out_dir: Path, tables_dir: Path) -> None:
        """Validate-on-close: every spec deliverable is present (preflight-pattern §4)."""
        required = [
            out_dir / "decision.json",
            out_dir / "report.md",
            tables_dir / "reconstruction_metrics.csv",
            tables_dir / "latent_stats.csv",
            tables_dir / "voided_tests.csv",
            tables_dir / "voided_roundtrip.csv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise MaisiAuditError(f"missing deliverables after run: {missing}")

    def _update_latest_symlink(self, output_root: Path, timestamp: str) -> None:
        latest = output_root / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
        except OSError:
            shutil.rmtree(latest, ignore_errors=True)
        try:
            os.symlink(timestamp, latest)  # relative target keeps the tree portable
        except OSError as exc:
            logger.warning("could not create LATEST symlink at %s: %s", latest, exc)


__all__ = ["PRODUCER_ID", "MaisiVaeEngine", "MaisiVaeRoutineConfig"]


# ---------------------------------------------------------------------------
# Local helpers (no public export)
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _json_num(x: float, ndigits: int = 4) -> float | None:
    """Round a float for JSON; non-finite values become ``None`` (JSON has no NaN)."""
    return round(float(x), ndigits) if np.isfinite(x) else None


def _csv_cell(value: object) -> object:
    """Format a metric value for a CSV cell (6 significant figures for floats)."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _summary(values: list[float]) -> dict[str, float]:
    """Mean / median / p10 / p90 over the finite entries of ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }
