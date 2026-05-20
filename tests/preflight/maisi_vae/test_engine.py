"""Engine-level tests for pre-flight 03 with a mock MAISI VAE.

The :class:`_FakeMaisiVAE` double runs on the CPU with no checkpoint, so the
full ``MaisiVaeEngine.run()`` flow — cohort selection, single-encode audit,
decision rule, artifact writing, validate-on-close, hard-fail gate — is
exercised without GPU or the real MAISI weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from routines.preflights.maisi_vae.engine import MaisiVaeEngine, MaisiVaeRoutineConfig

from brainrepa_fm.data.exceptions import MaisiAuditError, PreflightError

pytestmark = [pytest.mark.preflight_maisi, pytest.mark.unit]

_ENGINE_VAE = "routines.preflights.maisi_vae.engine.maisi_vae_engine.MaisiVAE"


class _FakeMaisiVAE:
    """Stateful test double for :class:`brainrepa_fm.common.maisi.MaisiVAE`.

    ``encode`` caches its input and returns a zero latent of the correct shape;
    ``decode`` returns the cached input plus optional Gaussian noise, so the
    round-trip PSNR is controllable (``noise_std=0`` → identity).
    """

    def __init__(self, *, noise_std: float = 0.0, channels: int = 4, factor: int = 4) -> None:
        self._noise_std = noise_std
        self._channels = channels
        self._factor = factor
        self._last: torch.Tensor | None = None
        self.scale_factor = 1.0
        self.info = SimpleNamespace(sha256_prefix="deadbeefdeadbeef")
        self._rng = np.random.default_rng(0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._last = x
        b, _, xx, yy, zz = x.shape
        return torch.zeros(
            (b, self._channels, xx // self._factor, yy // self._factor, zz // self._factor),
            dtype=torch.float32,
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # The cached encode input drives the round-trip; the latent is unused.
        del z
        assert self._last is not None
        out = self._last
        if self._noise_std > 0.0:
            noise = torch.from_numpy(
                self._rng.normal(0.0, self._noise_std, size=tuple(out.shape)).astype(np.float32)
            )
            out = (out + noise).clamp(0.0, 1.0)
        return out.to(torch.float32)


def _fake_factory(noise_std: float):
    """A patch target that ignores the engine's MaisiVAE kwargs."""
    return lambda **_kwargs: _FakeMaisiVAE(noise_std=noise_std)


def _smoke_cfg(h5_path: Path, out_root: Path, **overrides) -> MaisiVaeRoutineConfig:
    params: dict[str, object] = {
        "source_h5": h5_path,
        "output_root": out_root,
        "n_subjects": 2,
        "target_shape": "3060",
        "n_void_masks_per_volume": 2,
        "void_seed": 0,
        "seed": 0,
        "device": "cpu",
        "log_level": "WARNING",
    }
    params.update(overrides)
    return MaisiVaeRoutineConfig(**params)


def _build_empty_train_h5(path: Path) -> Path:
    """A minimal H5 whose ``splits/train ∩ gt/scan_index`` is empty."""
    with h5py.File(path, "w") as f:
        f.create_dataset("splits/train", data=np.array([], dtype=np.int32))
        f.create_dataset("gt/scan_index", data=np.array([0, 1], dtype=np.int32))
    return path


def test_config_loads_from_tiny_h5(tiny_brats_h5, tmp_path):
    cfg = _smoke_cfg(tiny_brats_h5, tmp_path / "out")
    assert cfg.target_shape == "3060"
    assert cfg.n_subjects == 2


def test_config_missing_h5_raises(tmp_path):
    with pytest.raises(ValueError):
        _smoke_cfg(tmp_path / "absent.h5", tmp_path / "out")


def test_config_rejects_zero_subjects(tiny_brats_h5, tmp_path):
    with pytest.raises(ValueError):
        _smoke_cfg(tiny_brats_h5, tmp_path / "out", n_subjects=0)


def test_engine_run_writes_all_deliverables(tiny_brats_h5, tmp_path):
    cfg = _smoke_cfg(tiny_brats_h5, tmp_path / "out")
    with patch(_ENGINE_VAE, _fake_factory(noise_std=1e-3)):
        out_dir = MaisiVaeEngine(cfg).run()

    assert (out_dir / "report.md").exists()
    assert (out_dir / "decision.json").exists()
    for name in ("reconstruction_metrics.csv", "latent_stats.csv", "voided_tests.csv"):
        assert (out_dir / "tables" / name).exists()
    assert list((out_dir / "figures").glob("*.png"))

    decision = json.loads((out_dir / "decision.json").read_text())
    assert decision["schema_version"] == "1.0"
    assert decision["path"] == "1"
    assert decision["vae_fine_tune"] is False
    assert decision["fine_tune_target"] == "none"
    assert decision["latent_aug_safe"] == []
    assert len(decision["latent_scale"]) == 4
    assert len(decision["latent_mean"]) == 4
    assert decision["n_volumes_audited"] == 2
    assert decision["target_shape"] == "3060"


def test_engine_creates_latest_symlink(tiny_brats_h5, tmp_path):
    cfg = _smoke_cfg(tiny_brats_h5, tmp_path / "out")
    with patch(_ENGINE_VAE, _fake_factory(noise_std=1e-3)):
        out_dir = MaisiVaeEngine(cfg).run()
    latest = cfg.output_root / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == out_dir.resolve()


def test_engine_path3_hard_fails_after_writing_artifacts(tiny_brats_h5, tmp_path):
    cfg = _smoke_cfg(tiny_brats_h5, tmp_path / "out")
    with patch(_ENGINE_VAE, _fake_factory(noise_std=0.15)):
        with pytest.raises(PreflightError):
            MaisiVaeEngine(cfg).run()
    stamps = [
        p for p in cfg.output_root.iterdir() if p.is_dir() and not p.is_symlink()
    ]
    assert stamps, "artifact directory must exist even on hard-fail"
    decision = json.loads((stamps[0] / "decision.json").read_text())
    assert decision["path"] == "3"
    assert decision["vae_fine_tune"] is True


def test_engine_empty_cohort_raises(tmp_path):
    h5_path = _build_empty_train_h5(tmp_path / "empty.h5")
    cfg = _smoke_cfg(h5_path, tmp_path / "out")
    with patch(_ENGINE_VAE, _fake_factory(noise_std=0.0)):
        with pytest.raises(MaisiAuditError):
            MaisiVaeEngine(cfg).run()
