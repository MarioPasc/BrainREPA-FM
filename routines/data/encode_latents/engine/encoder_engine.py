"""Thin YAML → :class:`LatentEncoder` orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from brainrepa_fm.data.latent_encoder import LatentEncodeConfig, LatentEncoder

logger = logging.getLogger(__name__)


class LatentEncoderEngine:
    """YAML-driven orchestrator for the latent encoder."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        with self.config_path.open("r") as fh:
            raw = yaml.safe_load(fh) or {}
        self.config = LatentEncodeConfig(**raw)

    def run(self) -> Path:
        logger.info("config: %s", self.config.model_dump())
        return LatentEncoder(self.config).run()


__all__ = ["LatentEncoderEngine"]
