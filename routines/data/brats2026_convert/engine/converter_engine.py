"""Thin orchestrator: load YAML → :class:`BraTS2026Converter` → validate → return path.

This module is a pure wire-up. Library logic lives in
:mod:`brainrepa_fm.data.brats2026_converter`. The engine does:

1. Parse the YAML config into :class:`BraTS2026ConvertConfig`.
2. Call :meth:`BraTS2026Converter.run`.
3. Return the validated H5 path.

Per ``.claude/rules/preflight-pattern.md`` item 2: a single ``run()`` method
returns the produced artifact path. No heavy work at import time (item 6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from brainrepa_fm.data.brats2026_converter import (
    BraTS2026ConvertConfig,
    BraTS2026Converter,
)

logger = logging.getLogger(__name__)


class BraTS2026ConvertEngine:
    """YAML-driven orchestrator around :class:`BraTS2026Converter`."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        with self.config_path.open("r") as fh:
            raw = yaml.safe_load(fh) or {}
        self.config = BraTS2026ConvertConfig(**raw)

    def run(self) -> Path:
        """Run the converter and return the validated H5 path."""
        logger.info("loading config from %s", self.config_path)
        logger.info("config: %s", self.config.model_dump())
        converter = BraTS2026Converter(self.config)
        return converter.run()


__all__ = ["BraTS2026ConvertEngine"]
