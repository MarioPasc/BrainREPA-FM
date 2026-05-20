"""Typed exceptions for the H5 producers and validators.

Per ``.claude/rules/coding-standards.md`` item 12, library code raises a typed
exception per module rather than bare ``Exception``.
"""

from __future__ import annotations


class H5SchemaError(Exception):
    """Base class for any HDF5 schema violation."""


class BratsH5SchemaError(H5SchemaError):
    """Schema A (`brats_inpainting_2026.h5`) violations."""


class LatentsH5SchemaError(H5SchemaError):
    """Schema B (`brainrepa_latents.h5`) violations."""


class ConverterError(Exception):
    """Raised by the NIfTI→H5 converter for source-data or I/O problems."""


class PreflightError(Exception):
    """Raised by pre-flight routines when a hard-fail gate trips."""


class MaisiAuditError(Exception):
    """Raised by the MAISI VAE audit (pre-flight 03) when an invariant is
    violated — e.g. an empty cohort, an all-NaN metric column, or a missing
    deliverable at validate-on-close. Distinct from :class:`PreflightError`,
    which signals a *decided* hard-fail gate (Path 3)."""
