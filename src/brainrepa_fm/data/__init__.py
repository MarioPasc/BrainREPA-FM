"""Data layer for BrainREPA-FM.

Owns the two H5 schemas (source `brats_inpainting_2026.h5` and latent
`brainrepa_latents.h5`), their validators, the NIfTI→H5 converter, and the
patient-level partitioner. Library-level — importable without invoking any CLI.

Every H5 producer in this package conforms to ``.claude/rules/h5-design-principles.md``:
attrs-driven, validator-paired, write-then-assert, gzip-4, self-describing.
"""

from brainrepa_fm.data.exceptions import (
    BratsH5SchemaError,
    H5SchemaError,
    LatentsH5SchemaError,
)

__all__ = [
    "BratsH5SchemaError",
    "H5SchemaError",
    "LatentsH5SchemaError",
]
