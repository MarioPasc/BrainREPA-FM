"""Patient-level partitioning for the BraTS-2026 training pool.

BraTS scan IDs follow the convention ``BraTS-GLI-NNNNN-XXX`` where ``NNNNN`` is
the patient identifier and ``XXX`` is a per-patient session index. A single
patient may contribute multiple scans (different sessions). To prevent
within-patient leakage between train, val, and test splits, partitioning groups
scans by their ``NNNNN`` patient identifier and assigns whole patients to a
single split deterministically (seeded).

The 219 challenge-validation subjects are external to this partition — they
land in their own ``challenge_val`` split unconditionally.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

_SCAN_ID_RE = re.compile(r"^BraTS-(?P<cohort>[A-Z]+)-(?P<patient>\d+)-(?P<session>\d+)$")


def extract_patient_id(scan_id: str) -> str:
    """Extract the ``NNNNN`` patient field from a BraTS scan ID.

    Parameters:
        scan_id: A string of the form ``BraTS-GLI-NNNNN-XXX``.

    Returns:
        The patient field as a 5-digit string.

    Raises:
        ValueError: If ``scan_id`` does not match the expected pattern.
    """
    match = _SCAN_ID_RE.match(scan_id)
    if match is None:
        raise ValueError(f"scan_id does not match BraTS-<cohort>-NNNNN-XXX: {scan_id!r}")
    return match.group("patient")


def partition_patients(
    scan_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Partition a list of scan IDs into ``{train, val, test}`` at the patient level.

    All scans of one patient land in exactly one split. The split sizes target the
    requested fractions in expectation; the actual sizes depend on per-patient
    scan counts. Deterministic given ``seed``.

    Parameters:
        scan_ids: Iterable of scan IDs to partition. Order is preserved in the
            returned int32 index arrays.
        fractions: ``(train, val, test)`` weights. Must sum to 1.0 within ``1e-6``.
        seed: Random seed for the patient-level shuffle.

    Returns:
        Mapping from split name (``"train"`` / ``"val"`` / ``"test"``) to an
        int32 array of indices into ``scan_ids``.

    Raises:
        ValueError: On malformed ``scan_ids`` or invalid ``fractions``.
    """
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must sum to 1.0; got {fractions} (sum={sum(fractions)})")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative; got {fractions}")

    by_patient: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(scan_ids):
        by_patient[extract_patient_id(sid)].append(i)

    patients = sorted(by_patient.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    patients_perm = [patients[i] for i in perm]

    n_patients = len(patients_perm)
    n_train = int(round(fractions[0] * n_patients))
    n_val = int(round(fractions[1] * n_patients))
    n_test = n_patients - n_train - n_val
    if n_test < 0:
        # Rare rounding-induced negative; steal from train.
        n_train += n_test
        n_test = 0

    train_p = patients_perm[:n_train]
    val_p = patients_perm[n_train : n_train + n_val]
    test_p = patients_perm[n_train + n_val :]

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for split_name, plist in (("train", train_p), ("val", val_p), ("test", test_p)):
        for p in plist:
            splits[split_name].extend(by_patient[p])

    return {name: np.array(sorted(idx), dtype=np.int32) for name, idx in splits.items()}


__all__ = ["extract_patient_id", "partition_patients"]
