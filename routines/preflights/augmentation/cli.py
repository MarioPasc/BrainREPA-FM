"""CLI: ``brainrepa-preflight-augmentation <config.yaml>``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

from routines.preflights.augmentation.engine import (
    AugmentationEngine,
    AugmentationRoutineConfig,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: brainrepa-preflight-augmentation <config.yaml>\n")
        return 2

    cfg_path = Path(args[0]).expanduser().resolve()
    if not cfg_path.exists():
        sys.stderr.write(f"config not found: {cfg_path}\n")
        return 2

    with cfg_path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = AugmentationRoutineConfig(**raw)

    logging.basicConfig(
        level=cfg.log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    engine = AugmentationEngine(cfg)
    out_dir = engine.run()
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
