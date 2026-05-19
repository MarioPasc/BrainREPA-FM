"""Schema A (source H5) → Schema B (latent H5) encoder.

Streams the source ``brats_inpainting_2026.h5`` scan by scan, encodes each
``images/t1_voided`` and (when present) each ``gt/t1`` through the frozen
MAISI-V2 VAE-GAN wrapper, and writes the two anchor latents to a Schema-B H5.

The CSR-augmented latents are NOT populated by this producer — they are filled
in by a downstream task once pre-flight 01's ``decision.json::include`` is
finalized. This producer writes empty placeholders for those groups so the
file remains schema-conformant.

Per ``.claude/rules/h5-design-principles.md``:
- gzip-4 compression on bulk latents.
- chunks ``(1, C, Lz, Ly, Lx)`` so one read = one scan.
- ``assert_brainrepa_latents_valid`` runs before atomic rename.
"""

from __future__ import annotations

import json
import logging
import os
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
    center_crop_to_maisi,
    probe_latent_shape,
    volume_to_tensor,
)
from brainrepa_fm.data.brainrepa_latents_schema import assert_brainrepa_latents_valid
from brainrepa_fm.data.h5_schemas import LATENTS_SCHEMA_VERSION

logger = logging.getLogger(__name__)

PRODUCER_ID: str = "routines.data.encode_latents:v0.0.1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LatentEncodeConfig(BaseModel):
    """Configuration for the Schema-A → Schema-B encoder.

    Attributes:
        source_h5: Schema-A H5 (read).
        output_path: Schema-B H5 to produce.
        target_shape: ``"3060"`` (192×192×144) or ``"a100"`` (256×256×192).
        max_subjects: Cap the number of scans encoded. None = all.
        device: Torch device for the VAE.
        gzip_level: HDF5 gzip level for bulk latents.
        log_level: Logging verbosity.
    """

    source_h5: Path
    output_path: Path
    target_shape: Literal["3060", "a100"] = "3060"
    max_subjects: int | None = None
    device: str = "cuda"
    gzip_level: int = Field(default=4, ge=0, le=9)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("source_h5")
    @classmethod
    def _resolve_source(cls, v: Path) -> Path:
        p = Path(v).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"source H5 not found: {p}")
        return p

    @field_validator("output_path")
    @classmethod
    def _resolve_output(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class LatentEncoder:
    """Producer for ``brainrepa_latents.h5`` (Schema B)."""

    def __init__(self, config: LatentEncodeConfig) -> None:
        self.config = config
        logging.basicConfig(level=config.log_level)
        logger.setLevel(config.log_level)

    def run(self) -> Path:
        cfg = self.config
        target_shape = MAISI_PAD_SHAPE_3060 if cfg.target_shape == "3060" else MAISI_PAD_SHAPE

        logger.info("loading MAISI VAE on %s", cfg.device)
        vae = MaisiVAE(device=cfg.device, autocast_fp16=True, use_checkpointing=True)

        # Probe latent shape once.
        latent_c, latent_z, latent_y, latent_x = probe_latent_shape(
            vae, input_shape=target_shape, device=cfg.device
        )
        latent_trailing = (latent_c, latent_z, latent_y, latent_x)
        logger.info("latent shape pinned: %s for input %s", latent_trailing, target_shape)

        with h5py.File(cfg.source_h5, "r") as src:
            n_scans = int(src.attrs["n_scans"])
            n_eff = n_scans if cfg.max_subjects is None else min(n_scans, cfg.max_subjects)
            gt_scan_index_src = (
                src["gt/scan_index"][...].astype(np.int32)
                if "gt/scan_index" in src
                else np.array([], dtype=np.int32)
            )
            scan_ids = np.array(
                [s.decode() if isinstance(s, bytes) else s for s in src["scan_id"][...]]
            )
            splits_arr = np.array(
                [s.decode() if isinstance(s, bytes) else s for s in src["split"][...]]
            )

            # When max_subjects is set, restrict to the first n_eff and intersect gt index.
            gt_to_global = {int(g): i for i, g in enumerate(gt_scan_index_src)}
            kept_gt_pairs: list[tuple[int, int]] = []  # (out_row_in_n_eff, src_gt_row)
            for out_row, src_row in enumerate(range(n_eff)):
                if src_row in gt_to_global:
                    kept_gt_pairs.append((out_row, gt_to_global[src_row]))
            n_with_gt = len(kept_gt_pairs)

            out_path = cfg.output_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            partial = out_path.with_suffix(out_path.suffix + ".partial")
            if partial.exists():
                partial.unlink()

            try:
                with h5py.File(partial, "w") as dst:
                    self._write_attrs(
                        dst,
                        n_scans=n_eff,
                        n_with_gt=n_with_gt,
                        latent_trailing=latent_trailing,
                        vae=vae,
                        source_h5=cfg.source_h5,
                    )
                    vlen = h5py.string_dtype(encoding="utf-8")
                    dst.create_dataset(
                        "scan_id",
                        data=np.asarray(scan_ids[:n_eff], dtype=object),
                        dtype=vlen,
                    )
                    dst.create_dataset(
                        "split",
                        data=np.asarray(splits_arr[:n_eff], dtype=object),
                        dtype=vlen,
                    )

                    chunk = (1, *latent_trailing)
                    voided_ds = dst.create_dataset(
                        "latents/voided_anchor",
                        shape=(n_eff, *latent_trailing),
                        dtype=np.float32,
                        chunks=chunk,
                        compression="gzip",
                        compression_opts=cfg.gzip_level,
                    )
                    gt_ds = dst.create_dataset(
                        "latents/gt_anchor",
                        shape=(n_with_gt, *latent_trailing),
                        dtype=np.float32,
                        chunks=chunk if n_with_gt > 0 else None,
                        compression="gzip" if n_with_gt > 0 else None,
                        compression_opts=cfg.gzip_level if n_with_gt > 0 else None,
                    )
                    dst.create_dataset(
                        "gt/scan_index",
                        data=np.array([p[0] for p in kept_gt_pairs], dtype=np.int32),
                    )

                    # Empty placeholders for the augmented CSR group (filled later).
                    dst.create_dataset(
                        "latents/augmented/values",
                        shape=(0, *latent_trailing),
                        maxshape=(None, *latent_trailing),
                        dtype=np.float32,
                        chunks=chunk,
                        compression="gzip",
                        compression_opts=cfg.gzip_level,
                    )
                    dst.create_dataset(
                        "latents/augmented/offsets",
                        data=np.zeros(n_eff + 1, dtype=np.int32),
                    )
                    dst.create_dataset(
                        "latents/augmented/augmentation_ids",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=vlen,
                    )
                    dst.create_dataset(
                        "augmentations/include", shape=(0,), maxshape=(None,), dtype=vlen
                    )

                    # Latent stats placeholders.
                    dst.create_dataset("latent_scale", data=np.zeros(latent_c, dtype=np.float32))
                    dst.create_dataset("latent_mean", data=np.zeros(latent_c, dtype=np.float32))

                    # Splits — restrict the source's index arrays to [0, n_eff).
                    for name in ("train", "val", "test", "challenge_val"):
                        src_idx = src[f"splits/{name}"][...].astype(np.int32)
                        if n_eff < n_scans:
                            src_idx = src_idx[src_idx < n_eff]
                        dst.create_dataset(f"splits/{name}", data=src_idx)

                    # Stream encode.
                    for out_row in range(n_eff):
                        sid = scan_ids[out_row]
                        t1v = src["images/t1_voided"][out_row]
                        z_voided = self._encode_one(vae, t1v, target_shape, cfg.device)
                        voided_ds[out_row] = z_voided
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(
                            "encoded voided %d/%d (%s)  ‖z‖=%.3f",
                            out_row + 1,
                            n_eff,
                            sid,
                            float(np.linalg.norm(z_voided)),
                        )

                    for out_row, src_gt_row in kept_gt_pairs:
                        sid = scan_ids[out_row]
                        gt_t1 = src["gt/t1"][src_gt_row]
                        z_gt = self._encode_one(vae, gt_t1, target_shape, cfg.device)
                        gt_ds[kept_gt_pairs.index((out_row, src_gt_row))] = z_gt
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(
                            "encoded gt %s ‖z‖=%.3f",
                            sid,
                            float(np.linalg.norm(z_gt)),
                        )

                    self._add_dataset_attrs(dst)

                assert_brainrepa_latents_valid(partial)
            except Exception:
                if partial.exists():
                    partial.unlink(missing_ok=True)
                raise

            os.replace(partial, out_path)
            logger.info("wrote and validated %s", out_path)
            return out_path

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _encode_one(
        vae: MaisiVAE, vol: np.ndarray, target_shape: tuple[int, int, int], device: str
    ) -> np.ndarray:
        t = volume_to_tensor(vol, device=device)
        t_crop, _ = center_crop_to_maisi(t, target_shape=target_shape)
        z = vae.encode(t_crop)
        z_np = z.detach().to(dtype=torch.float32, device="cpu").numpy()
        return z_np[0]  # (C, Lz, Ly, Lx)

    def _write_attrs(
        self,
        dst: h5py.File,
        *,
        n_scans: int,
        n_with_gt: int,
        latent_trailing: tuple[int, ...],
        vae: MaisiVAE,
        source_h5: Path,
    ) -> None:
        c, lz, ly, lx = latent_trailing
        dst.attrs["schema_version"] = LATENTS_SCHEMA_VERSION
        dst.attrs["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dst.attrs["producer"] = PRODUCER_ID
        dst.attrs["config_json"] = json.dumps(asdict_cfg(self.config), sort_keys=True)
        dst.attrs["git_sha"] = _git_sha()
        dst.attrs["n_scans"] = n_scans
        dst.attrs["n_with_gt"] = n_with_gt
        dst.attrs["latent_stats_calibrated"] = False
        dst.attrs["vae_checkpoint_sha256"] = vae.info.sha256_prefix
        dst.attrs["vae_scale_factor"] = float(vae.scale_factor)
        dst.attrs["paired_source"] = str(source_h5)
        dst.attrs["latent_channels"] = c
        dst.attrs["latent_spatial_shape"] = json.dumps([lz, ly, lx])

    @staticmethod
    def _add_dataset_attrs(dst: h5py.File) -> None:
        from brainrepa_fm.data.h5_schemas import LATENTS_SCHEMA

        for spec in LATENTS_SCHEMA.datasets:
            if spec.path not in dst:
                continue
            d = dst[spec.path]
            d.attrs["units"] = spec.units
            d.attrs["description"] = spec.description
            d.attrs["dtype"] = spec.dtype
            d.attrs["leading_dim"] = spec.leading_dim


def asdict_cfg(cfg: LatentEncodeConfig) -> dict[str, object]:
    d = cfg.model_dump()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
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


__all__ = ["PRODUCER_ID", "LatentEncodeConfig", "LatentEncoder"]
