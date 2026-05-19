"""CLI: ``brainrepa-data-brats2026-convert <config.yaml>``.

Per ``.claude/rules/preflight-pattern.md`` item 1: one positional argument
(the YAML path). No other flags. Logging level is read from the YAML.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.data.brats2026_convert.engine import BraTS2026ConvertEngine


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``brainrepa-data-brats2026-convert`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: brainrepa-data-brats2026-convert <config.yaml>\n")
        return 2
    config_path = Path(args[0])

    # Defer log-level resolution to the engine config; baseline at INFO.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    engine = BraTS2026ConvertEngine(config_path)
    logging.getLogger().setLevel(engine.config.log_level)

    out_path = engine.run()
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
