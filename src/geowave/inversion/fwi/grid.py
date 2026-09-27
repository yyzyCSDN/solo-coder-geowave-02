from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.ray.bending import GridVelocity


@dataclass(frozen=True)
class GridSpec:
    """Regular 2-D computational grid. x is horizontal, z points downward."""

    x: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'x', np.asarray(self.x, dtype=float))
        object.__setattr__(self, 'z', np.asarray(self.z, dtype=float))
        if self.x.ndim != 1 or self.z.ndim != 1 or self.x.size < 5 or self.z.size < 5:
            raise ValueError('grid axes must be 1-D with at least 5 samples')
        if np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.z) <= 0):
            raise ValueError('grid axes must increase strictly')

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dz(self) -> float:
        return float(self.z[1] - self.z[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z.size, self.x.size)

    @property
    def nx(self) -> int:
        return self.x.size

    @property
    def nz(self) -> int:
        return self.z.size

    def regular(self, rtol: float = 1e-6) -> bool:
        return bool(np.allclose(np.diff(self.x), self.dx, rtol=rtol)
                    and np.allclose(np.diff(self.z), self.dz, rtol=rtol))

    def nearest(self, x: float, z: float) -> tuple[int, int]:
        ix = int(np.argmin(np.abs(self.x - x)))
        iz = int(np.argmin(np.abs(self.z - z)))
        return (iz, ix)

    def bilinear_weights(self, x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices and weights for bilinear interpolation at a physical point."""
        ix = int(np.clip(np.searchsorted(self.x, x) - 1, 0, self.nx - 2))
        iz = int(np.clip(np.searchsorted(self.z, z) - 1, 0, self.nz - 2))
        fx = (x - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        fz = (z - self.z[iz]) / (self.z[iz + 1] - self.z[iz])
        fx = float(np.clip(fx, 0.0, 1.0))
        fz = float(np.clip(fz, 0.0, 1.0))
        ids = np.array([(iz, ix), (iz, ix + 1), (iz + 1, ix), (iz + 1, ix + 1)])
        w = np.array([(1 - fx) * (1 - fz), fx * (1 - fz), (1 - fx) * fz, fx * fz])
        return (ids, w)


def stable_dt(spec: GridSpec, max_velocity_mps: float, cfl: float = 0.45) -> float:
    """Largest time step satisfying the 2-D acoustic CFL condition on the stencil."""
    h = min(spec.dx, spec.dz)
    return cfl * h / (max_velocity_mps * np.sqrt(2.0))


def as_grid_velocity(spec: GridSpec, velocity: np.ndarray) -> GridVelocity:
    v = np.asarray(velocity, dtype=float)
    if v.shape != spec.shape:
        raise ValueError(f'velocity shape {v.shape} != grid shape {spec.shape}')
    return GridVelocity(spec.x, spec.z, v)


def clamp_velocity(velocity: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
    v = np.asarray(velocity, dtype=float)
    lo = -np.inf if vmin is None else float(vmin)
    hi = np.inf if vmax is None else float(vmax)
    if lo > hi:
        raise ValueError(f'empty velocity range [{vmin}, {vmax}]')
    return np.clip(v, lo, hi)
