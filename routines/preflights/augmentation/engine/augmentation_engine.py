"""Pre-flight 01 engine — augmentation composability with the MAISI VAE.

Reads the source ``brats_inpainting_2026.h5`` (Schema A), picks a stratified
subset of training subjects, runs the eight transforms against the frozen VAE,
applies the decision rules from ``docs/checks/01_augmentation_preflight.md``
§3.4, performs the four-descriptor KS audit against the challenge-val
distribution, and writes the artifact bundle.

Outputs (per ``.claude/rules/preflight-pattern.md``):
``artifacts/preflights/augmentation/<UTC>/``
    ├── report.md
    ├── figures/{aug_<id>_<S>.png, ks_<descriptor>.png}
    ├── tables/{composition_gaps.csv, ks_results.csv}
    └── decision.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
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
    tensor_to_volume,
    volume_to_tensor,
)
from brainrepa_fm.data.exceptions import PreflightError
from brainrepa_fm.preflight.augmentation.mask_audit import (
    HARD_FAIL_DESCRIPTORS,
    MASK_DESCRIPTORS,
    compute_mask_descriptors,
    decide_hard_fail,
    ks_test,
)
from brainrepa_fm.preflight.augmentation.transforms import (
    ALL_TRANSFORMS,
    TransformSpec,
    apply_transform,
)
from brainrepa_fm.preflight.augmentation.vae_composability import (
    DeltaResult,
    aggregate_deltas,
    compute_delta_aug_vae,
)
from brainrepa_fm.preflight.augmentation.visualize import (
    render_ks_cdf,
    render_qc_grid,
)

logger = logging.getLogger(__name__)

# Producer identifier persisted into decision.json.
PRODUCER_ID: str = "routines.preflights.augmentation:v0.0.1"
DECISION_SCHEMA_VERSION: str = "1.0"

# Decision-rule thresholds (docs/checks/01_augmentation_preflight.md §3.4).
DELTA_INCLUDE_DB: float = 0.5
DELTA_HALVE_DB: float = 1.5
C4_SPECIAL_DELTA_DB: float = 0.3
C4_SPECIAL_LATENT_RATIO: float = 0.02


# ---------------------------------------------------------------------------
# Routine config
# ---------------------------------------------------------------------------


class AugmentationRoutineConfig(BaseModel):
    """Pydantic config for the augmentation pre-flight."""

    source_h5: Path
    output_root: Path = Path("artifacts/preflights/augmentation")
    n_train_subjects: int = Field(default=50, ge=2)
    n_val_subjects: int = Field(default=219, ge=2)
    stratify_by: Literal["random", "tumor_volume_quartile"] = "tumor_volume_quartile"
    target_shape: Literal["3060", "a100"] = "3060"
    device: str = "cuda"
    seed: int = 2026
    n_figure_subjects: int = Field(default=2, ge=1, le=4)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AugmentationEngine:
    """YAML-driven orchestrator for pre-flight 01."""

    def __init__(self, config: AugmentationRoutineConfig) -> None:
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

        target_shape = MAISI_PAD_SHAPE_3060 if cfg.target_shape == "3060" else MAISI_PAD_SHAPE

        with h5py.File(cfg.source_h5, "r") as f:
            train_idx = f["splits/train"][...].astype(int)
            chal_idx = f["splits/challenge_val"][...].astype(int)
            scan_ids = np.array(
                [s.decode() if isinstance(s, bytes) else s for s in f["scan_id"][...]]
            )

            # Map global -> gt-local idx so we can fetch tumor masks for donors.
            gt_scan_index = (
                f["gt/scan_index"][...].astype(int)
                if "gt/scan_index" in f
                else np.array([], dtype=int)
            )
            global_to_gt = {int(g): i for i, g in enumerate(gt_scan_index)}

            # Pick the auditing subset.
            train_subset = self._pick_train_subset(f, train_idx, cfg)
            val_subset = chal_idx[: min(cfg.n_val_subjects, len(chal_idx))]
            logger.info(
                "selected %d train scans + %d challenge-val scans",
                len(train_subset),
                len(val_subset),
            )

            # ---- step 1: instantiate VAE ---------------------------------
            logger.info("loading MAISI VAE (target_shape=%s)", target_shape)
            vae = MaisiVAE(device=cfg.device, autocast_fp16=True, use_checkpointing=True)

            # ---- step 2: per-transform per-scan Δ_aug-VAE -----------------
            all_results: list[DeltaResult] = []
            qc_subjects = train_subset[: cfg.n_figure_subjects]
            qc_labels = (["S★", "S†", "S‡", "S§"])[: len(qc_subjects)]

            for trans_i, transform in enumerate(ALL_TRANSFORMS):
                for s_pos, scan_idx in enumerate(train_subset):
                    scan_idx_i = int(scan_idx)
                    sid = scan_ids[scan_idx_i]
                    seed = int(cfg.seed) + 1000 * trans_i + s_pos
                    t1v = f["images/t1_voided"][scan_idx_i]
                    brain = f["masks/brain"][scan_idx_i]
                    void = f["masks/void"][scan_idx_i]

                    donor_tumor: np.ndarray | None = None
                    if transform.id == "A.3":
                        donor_tumor = self._pick_donor_tumor(
                            f, train_idx, exclude=scan_idx_i, seed=seed, global_to_gt=global_to_gt
                        )

                    res = compute_delta_aug_vae(
                        vae,
                        scan_id=sid,
                        t1_voided=t1v,
                        brain=brain,
                        void=void,
                        transform=transform,
                        donor_tumor=donor_tumor,
                        seed=seed,
                        use_halved=False,
                        target_shape=target_shape,
                        device=cfg.device,
                    )
                    all_results.append(res)
                    if (s_pos + 1) % 10 == 0 or s_pos + 1 == len(train_subset):
                        logger.info(
                            "[%s] scan %d/%d Δ_void = %.3f dB",
                            transform.id,
                            s_pos + 1,
                            len(train_subset),
                            res.delta["void"],
                        )

                    # QC figures for first n_figure_subjects subjects.
                    if scan_idx_i in qc_subjects:
                        pos = qc_subjects.index(scan_idx_i)
                        fig_path = figures_dir / f"aug_{transform.id}_{qc_labels[pos]}.png"
                        self._render_subject_figure(
                            vae,
                            transform=transform,
                            t1v=t1v,
                            brain=brain,
                            void=void,
                            donor_tumor=donor_tumor,
                            subject_label=qc_labels[pos],
                            target_shape=target_shape,
                            seed=seed,
                            out_path=fig_path,
                            device=cfg.device,
                        )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---- step 3: aggregate & decision -----------------------------
            agg = aggregate_deltas(all_results)
            include, drop, halve_range, drop_reasons = self._decide_rules(all_results, agg)

            # ---- step 4: KS audit on mask distributions -------------------
            train_descs, val_descs = self._collect_mask_descriptors(
                f, train_subset=train_subset, val_subset=val_subset, include=include
            )
            ks_p = ks_test(train_descs, val_descs)
            ks_hard_fail = decide_hard_fail(ks_p)

            # ---- step 5: write artifact bundle ----------------------------
            self._write_tables(tables_dir, all_results=all_results, agg=agg, ks_p=ks_p)
            self._write_ks_figures(
                figures_dir=figures_dir, train_descs=train_descs, val_descs=val_descs, ks_p=ks_p
            )
            decision = {
                "schema_version": DECISION_SCHEMA_VERSION,
                "include": include,
                "drop": drop,
                "drop_reasons": drop_reasons,
                "halve_range": halve_range,
                "vae_composition_gap_db": {tid: float(agg[tid]["void"]["median"]) for tid in agg},
                "ks_p_values": {k: (float(v) if np.isfinite(v) else None) for k, v in ks_p.items()},
                "ks_hard_fail": bool(ks_hard_fail),
            }
            (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
            self._write_report(
                out_dir=out_dir,
                agg=agg,
                decision=decision,
                train_subset=train_subset,
                val_subset=val_subset,
            )

            # ---- step 6: LATEST symlink ----------------------------------
            self._update_latest_symlink(cfg.output_root, ts)

            if ks_hard_fail:
                msg = (
                    f"KS hard-fail: descriptors {[d for d in HARD_FAIL_DESCRIPTORS if np.isfinite(ks_p.get(d, np.nan)) and ks_p[d] < 0.05]} "
                    f"crossed the p < 0.05 threshold"
                )
                logger.error("%s", msg)
                raise PreflightError(msg)

        logger.info("decision.json written: %s", out_dir / "decision.json")
        return out_dir

    # -- helpers ----------------------------------------------------------

    def _pick_train_subset(
        self, f: h5py.File, train_idx: np.ndarray, cfg: AugmentationRoutineConfig
    ) -> list[int]:
        n = min(cfg.n_train_subjects, len(train_idx))
        if cfg.stratify_by == "random":
            rng = np.random.default_rng(cfg.seed)
            picks = rng.choice(train_idx, size=n, replace=False)
            return sorted(int(i) for i in picks)

        # tumor_volume_quartile: bucket train scans by tumor volume and pick uniformly.
        gt_index = f["gt/scan_index"][...].astype(int)
        gt_pos = {int(g): i for i, g in enumerate(gt_index)}
        rng = np.random.default_rng(cfg.seed)
        tumor_volumes: list[tuple[int, float]] = []
        for g in train_idx:
            gi = gt_pos.get(int(g))
            if gi is None:
                continue
            vol = float(f["gt/tumor_mask"][gi].sum())
            tumor_volumes.append((int(g), vol))
        if not tumor_volumes:
            return sorted(int(i) for i in rng.choice(train_idx, size=n, replace=False))
        tumor_volumes.sort(key=lambda x: x[1])
        # Sample evenly across quartiles.
        per_bucket = max(1, n // 4)
        chunks = np.array_split(tumor_volumes, 4)
        picks: list[int] = []
        for chunk in chunks:
            chunk_n = min(per_bucket, len(chunk))
            if chunk_n == 0:
                continue
            idx_choices = rng.choice(len(chunk), size=chunk_n, replace=False)
            for i in idx_choices:
                picks.append(int(chunk[i][0]))
        # Top up if we under-shot due to integer division.
        if len(picks) < n:
            remaining = [int(g) for g, _ in tumor_volumes if int(g) not in set(picks)]
            extra = rng.choice(remaining, size=min(n - len(picks), len(remaining)), replace=False)
            picks.extend(int(i) for i in extra)
        return sorted(picks[:n])

    def _pick_donor_tumor(
        self,
        f: h5py.File,
        train_idx: np.ndarray,
        *,
        exclude: int,
        seed: int,
        global_to_gt: dict[int, int],
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pool = [int(g) for g in train_idx if int(g) != exclude and int(g) in global_to_gt]
        donor_global = int(rng.choice(pool))
        donor_local = global_to_gt[donor_global]
        return f["gt/tumor_mask"][donor_local]

    def _render_subject_figure(
        self,
        vae: MaisiVAE,
        *,
        transform: TransformSpec,
        t1v: np.ndarray,
        brain: np.ndarray,
        void: np.ndarray,
        donor_tumor: np.ndarray | None,
        subject_label: str,
        target_shape: tuple[int, int, int],
        seed: int,
        out_path: Path,
        device: str,
    ) -> None:
        # Center-crop everything to envelope.
        from brainrepa_fm.preflight.augmentation.vae_composability import _crop_to_envelope

        t1_c, brain_c, void_c = _crop_to_envelope(t1v, brain, void, target_shape)
        donor_c = None
        if donor_tumor is not None:
            donor_c, _, _ = _crop_to_envelope(donor_tumor, brain, void, target_shape)
        t1_aug, void_aug, brain_aug = apply_transform(
            transform,
            t1_voided=t1_c,
            brain=brain_c,
            void=void_c,
            donor_tumor=donor_c,
            seed=seed,
        )
        t = volume_to_tensor(t1_aug, device=device)
        decoded = tensor_to_volume(vae.encode_decode(t).float())
        render_qc_grid(
            t1_baseline=t1_c,
            t1_aug=t1_aug,
            decoded_aug=decoded,
            void_aug=void_aug,
            brain=brain_aug,
            transform_id=transform.id,
            subject_label=subject_label,
            out_path=out_path,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _collect_mask_descriptors(
        self,
        f: h5py.File,
        *,
        train_subset: list[int],
        val_subset: np.ndarray,
        include: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Descriptor distributions for the KS test."""
        train_vals: dict[str, list[float]] = {n: [] for n in MASK_DESCRIPTORS}
        val_vals: dict[str, list[float]] = {n: [] for n in MASK_DESCRIPTORS}

        for scan_idx in train_subset:
            brain = f["masks/brain"][int(scan_idx)]
            for tid in include:
                seed = int(scan_idx) * 100 + hash(tid) % 100
                transform = next(t for t in ALL_TRANSFORMS if t.id == tid)
                # We only audit void-mask transforms here; intensity transforms keep the void.
                if transform.kind == "void_mask":
                    if transform.id == "A.3":
                        donor = self._pick_donor_tumor(
                            f,
                            np.asarray([int(s) for s in train_subset], dtype=int),
                            exclude=int(scan_idx),
                            seed=seed,
                            global_to_gt={int(g): i for i, g in enumerate(f["gt/scan_index"][...])},
                        )
                        new_void = self._sample_for_descriptor(transform, brain, donor, seed)
                    else:
                        new_void = self._sample_for_descriptor(transform, brain, None, seed)
                else:
                    # Spatial / intensity transforms — descriptors taken from the unchanged void.
                    new_void = f["masks/void"][int(scan_idx)]
                desc = compute_mask_descriptors(new_void, brain)
                for name in MASK_DESCRIPTORS:
                    train_vals[name].append(desc[name])

        for scan_idx in val_subset:
            brain = f["masks/brain"][int(scan_idx)]
            void = f["masks/void"][int(scan_idx)]
            desc = compute_mask_descriptors(void, brain)
            for name in MASK_DESCRIPTORS:
                val_vals[name].append(desc[name])

        return (
            {n: np.array(v, dtype=np.float64) for n, v in train_vals.items()},
            {n: np.array(v, dtype=np.float64) for n, v in val_vals.items()},
        )

    @staticmethod
    def _sample_for_descriptor(
        transform: TransformSpec,
        brain: np.ndarray,
        donor: np.ndarray | None,
        seed: int,
    ) -> np.ndarray:
        from brainrepa_fm.preflight.augmentation.transforms import (
            sample_donor_tumor_mask,
            sample_void_mask,
        )

        if transform.id == "A.3":
            assert donor is not None
            return sample_donor_tumor_mask(donor, brain, seed=seed)
        widen = float(transform.params.get("widen_factor", 1.0))
        return sample_void_mask(brain, tumor=None, widen_factor=widen, seed=seed)

    def _decide_rules(
        self, results: list[DeltaResult], agg: dict[str, dict[str, dict[str, float]]]
    ) -> tuple[list[str], list[str], list[str], dict[str, str]]:
        include: list[str] = []
        drop: list[str] = []
        halve: list[str] = []
        drop_reasons: dict[str, str] = {}

        # Per-transform mean latent_ratio (used by C.4 special rule).
        latent_ratios: dict[str, list[float]] = {}
        for r in results:
            latent_ratios.setdefault(r.transform_id, []).append(r.latent_ratio)

        for tid, stats in agg.items():
            void_median = float(stats["void"]["median"])
            mean_latent = float(np.mean(latent_ratios.get(tid, [np.nan])))

            if (
                tid == "C.4"
                and void_median < C4_SPECIAL_DELTA_DB
                and mean_latent < C4_SPECIAL_LATENT_RATIO
            ):
                drop.append(tid)
                drop_reasons[tid] = "VAE_erased_noise"
                continue

            if void_median <= DELTA_INCLUDE_DB:
                include.append(tid)
            elif void_median <= DELTA_HALVE_DB:
                include.append(tid)
                halve.append(tid)
            else:
                drop.append(tid)
                drop_reasons[tid] = (
                    f"void Δ_aug-VAE median = {void_median:.2f} dB > {DELTA_HALVE_DB} dB"
                )
        return include, drop, halve, drop_reasons

    def _write_tables(
        self,
        tables_dir: Path,
        *,
        all_results: list[DeltaResult],
        agg: dict[str, dict[str, dict[str, float]]],
        ks_p: dict[str, float],
    ) -> None:
        # composition_gaps.csv: one row per (transform, scan, region)
        comp_path = tables_dir / "composition_gaps.csv"
        with comp_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "transform_id",
                    "scan_id",
                    "region",
                    "baseline_psnr",
                    "aug_psnr",
                    "delta_db",
                    "latent_ratio",
                ]
            )
            for r in all_results:
                for region in r.delta:
                    writer.writerow(
                        [
                            r.transform_id,
                            r.scan_id,
                            region,
                            f"{r.baseline_psnr[region]:.4f}",
                            f"{r.aug_psnr[region]:.4f}",
                            f"{r.delta[region]:.4f}",
                            f"{r.latent_ratio:.6f}",
                        ]
                    )

        agg_path = tables_dir / "composition_gaps_aggregated.csv"
        with agg_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["transform_id", "region", "mean", "median", "p10", "p90"])
            for tid, regions in agg.items():
                for region, stats in regions.items():
                    writer.writerow(
                        [
                            tid,
                            region,
                            f"{stats['mean']:.4f}",
                            f"{stats['median']:.4f}",
                            f"{stats['p10']:.4f}",
                            f"{stats['p90']:.4f}",
                        ]
                    )

        ks_path = tables_dir / "ks_results.csv"
        with ks_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["descriptor", "p_value", "hard_fail_descriptor"])
            for name in MASK_DESCRIPTORS:
                p = ks_p.get(name, float("nan"))
                writer.writerow(
                    [
                        name,
                        f"{p:.6e}" if np.isfinite(p) else "nan",
                        "yes" if name in HARD_FAIL_DESCRIPTORS else "no",
                    ]
                )

    def _write_ks_figures(
        self,
        *,
        figures_dir: Path,
        train_descs: dict[str, np.ndarray],
        val_descs: dict[str, np.ndarray],
        ks_p: dict[str, float],
    ) -> None:
        for name in MASK_DESCRIPTORS:
            render_ks_cdf(
                train_descs[name],
                val_descs[name],
                descriptor=name,
                p_value=ks_p[name],
                out_path=figures_dir / f"ks_{name}.png",
            )

    def _write_report(
        self,
        *,
        out_dir: Path,
        agg: dict[str, dict[str, dict[str, float]]],
        decision: dict[str, object],
        train_subset: list[int],
        val_subset: np.ndarray,
    ) -> None:
        lines = [
            "# Pre-flight 01 — augmentation composability",
            "",
            f"_Generated at {datetime.now(timezone.utc).isoformat(timespec='seconds')} by {PRODUCER_ID}._",
            "",
            f"- N train subjects audited: {len(train_subset)}",
            f"- N val subjects in KS audit: {len(val_subset)}",
            f"- VAE envelope: {self.config.target_shape}",
            "",
            "## Per-transform Δ_aug-VAE (dB; positive = augmentation hurts VAE)",
            "",
            "| Transform | mean (void) | median (void) | p10 (void) | p90 (void) | median (brain) | median (full) |",
            "|---|---|---|---|---|---|---|",
        ]
        for tid, stats in agg.items():
            lines.append(
                f"| {tid} | "
                f"{stats['void']['mean']:.3f} | "
                f"{stats['void']['median']:.3f} | "
                f"{stats['void']['p10']:.3f} | "
                f"{stats['void']['p90']:.3f} | "
                f"{stats['brain']['median']:.3f} | "
                f"{stats['full']['median']:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"- **include**: {decision['include']}",
                f"- **halve_range**: {decision['halve_range']}",
                f"- **drop**: {decision['drop']}",
                f"- **drop_reasons**: {decision['drop_reasons']}",
                f"- **ks_p_values**: {decision['ks_p_values']}",
                f"- **ks_hard_fail**: {decision['ks_hard_fail']}",
                "",
                "## QC figures",
                "",
            ]
        )
        for fp in sorted((out_dir / "figures").glob("aug_*.png")):
            lines.append(f"![{fp.name}](figures/{fp.name})")
        lines.extend(
            [
                "",
                "## KS CDF figures",
                "",
            ]
        )
        for fp in sorted((out_dir / "figures").glob("ks_*.png")):
            lines.append(f"![{fp.name}](figures/{fp.name})")
        (out_dir / "report.md").write_text("\n".join(lines))

    def _update_latest_symlink(self, output_root: Path, timestamp: str) -> None:
        latest = output_root / "LATEST"
        # Use a relative symlink so the artifact tree is portable.
        target = timestamp
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
        except OSError:
            shutil.rmtree(latest, ignore_errors=True)
        try:
            os.symlink(target, latest)
        except OSError as exc:
            logger.warning("could not create LATEST symlink at %s: %s", latest, exc)


__all__ = ["PRODUCER_ID", "AugmentationEngine", "AugmentationRoutineConfig"]


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
