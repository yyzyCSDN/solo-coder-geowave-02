from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ShotGeometry:
    """One shot and the receivers that record it (2-D, x horizontal, z depth)."""

    shot_id: str
    source_xz: tuple[float, float]
    receiver_xz: np.ndarray   # (nr, 2)

    def __post_init__(self):
        rec = np.asarray(self.receiver_xz, dtype=float)
        if rec.ndim != 2 or rec.shape[1] != 2 or rec.shape[0] == 0:
            raise ValueError('receiver_xz must be a non-empty (nr, 2) array')
        object.__setattr__(self, 'receiver_xz', rec)

    @property
    def n_receivers(self) -> int:
        return self.receiver_xz.shape[0]


@dataclass(frozen=True)
class Survey:
    shots: tuple[ShotGeometry, ...]

    def __post_init__(self):
        if not self.shots:
            raise ValueError('survey contains no shots')

    def __iter__(self):
        return iter(self.shots)

    def __len__(self) -> int:
        return len(self.shots)

    def consistent(self) -> bool:
        nr = self.shots[0].n_receivers
        return all(s.n_receivers == nr for s in self.shots)


def surface_survey(source_x: np.ndarray, receiver_x: np.ndarray,
                   source_depth_m: float = 2.0, receiver_depth_m: float = 2.0) -> Survey:
    """Fixed-spread surface acquisition: every source is recorded by all receivers."""
    src_x = np.asarray(source_x, float)
    rec_x = np.asarray(receiver_x, float)
    recs = np.column_stack([rec_x, np.full_like(rec_x, receiver_depth_m)])
    shots = tuple(
        ShotGeometry(f'shot-{i:04d}', (float(sx), source_depth_m), recs)
        for i, sx in enumerate(src_x)
    )
    return Survey(shots)


@dataclass(frozen=True)
class SyntheticData:
    """Modeled common-shot gathers (observed data for an FWI exercise)."""

    velocity: np.ndarray
    shots: tuple[ShotGeometry, ...]
    gathers: np.ndarray   # (nshots, nt, nr)
    dt_s: float
    wavelet_name: str
