"""CLI: ``brainrepa-data-encode-latents <config.yaml>``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from routines.data.encode_latents.engine import LatentEncoderEngine


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: brainrepa-data-encode-latents <config.yaml>\n")
        return 2

    cfg_path = Path(args[0])
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    engine = LatentEncoderEngine(cfg_path)
    logging.getLogger().setLevel(engine.config.log_level)

    out_path = engine.run()
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
