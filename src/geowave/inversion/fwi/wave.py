from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.inversion.fwi.finite import assert_finite
from geowave.inversion.fwi.grid import GridSpec


@dataclass(frozen=True)
class AbsorbingBoundary:
    """Cerjan exponential damping frame.

    By default all four sides are damped. For land/near-surface acquisition set
    ``damp_top=False`` to keep the top as a reflecting free surface while only
    the sides and bottom absorb outgoing waves.
    """

    width: int = 30
    strength: float = 0.015
    damp_top: bool = True
    damp_bottom: bool = True
    damp_left: bool = True
    damp_right: bool = True

    def damping_1d(self, n: int, lo_side: bool = True, hi_side: bool = True) -> np.ndarray:
        """Per-step retention factor: exp(-s (depth_into_frame / width)^2)."""
        w = min(self.width, n // 2)
        d = np.ones(n)
        if w <= 0:
            return d
        if lo_side:
            d[:w] = np.exp(-self.strength * (np.arange(w, 0, -1) / w) ** 2)
        if hi_side:
            d[n - w:] = np.exp(-self.strength * (np.arange(1, w + 1) / w) ** 2)
        return d

    def damping(self, spec: GridSpec) -> np.ndarray:
        dz_1d = self.damping_1d(spec.nz, self.damp_top, self.damp_bottom)
        dx_1d = self.damping_1d(spec.nx, self.damp_left, self.damp_right)
        return np.outer(dz_1d, dx_1d)


@dataclass(frozen=True)
class ForwardResult:
    data: np.ndarray          # (nt, nr) receiver recordings
    snapshots: np.ndarray     # (nt, nz, nx) pressure field history
    dt_s: float


def _inject(field: np.ndarray, point_ids: np.ndarray, weights: np.ndarray, value: float,
            area: float) -> None:
    for (iz, ix), w in zip(point_ids, weights):
        field[iz, ix] += value * w * area


def _sample(field: np.ndarray, point_ids: np.ndarray, weights: np.ndarray) -> float:
    return float(sum(field[iz, ix] * w for (iz, ix), w in zip(point_ids, weights)))


def _laplacian(p: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """Symmetric 5-point Dirichlet Laplacian (zero ghosts on the outer border).

    Symmetry (L == L.T) is required so that reverse-time propagation is the
    exact discrete transpose of the forward operator.
    """
    g = np.pad(p, 1, mode='constant', constant_values=0.0)
    return (
        (g[1:-1, 2:] - 2.0 * g[1:-1, 1:-1] + g[1:-1, :-2]) / dx ** 2
        + (g[2:, 1:-1] - 2.0 * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dz ** 2
    )


def forward_model(velocity: np.ndarray, spec: GridSpec, dt_s: float, nt: int,
                  source: np.ndarray, src_xyz: tuple[float, float],
                  rec_xyz: np.ndarray, abc: AbsorbingBoundary | None = None,
                  *, store_wavefield: bool = True) -> ForwardResult:
    """Constant-density 2-D acoustic FD: p_tt = v^2 (p_xx + p_zz) + f(t) delta(x-xs).

    The source injects with the cell-area factor; the receiver samples the
    field without it, so the two operators are exact discrete transposes
    (required for the adjoint-state gradient).
    """
    v = np.asarray(velocity, dtype=float)
    source = np.asarray(source, dtype=float)
    if v.shape != spec.shape:
        raise ValueError('velocity shape does not match grid')
    if source.shape != (nt,):
        raise ValueError(f'source must have {nt} samples, got {source.shape}')
    assert_finite('velocity model', v)
    assert_finite('source wavelet', source)
    if np.any(v <= 0):
        raise ValueError('velocity must be positive everywhere')
    # CFL guard: reject an unstable time step before the field can blow up.
    cfl = float(np.max(v)) * np.sqrt(1.0 / spec.dx ** 2 + 1.0 / spec.dz ** 2) * dt_s
    if cfl >= 1.0:
        raise ValueError(
            f'time step unstable: CFL number {cfl:.3f} >= 1; use dt <= '
            f'{min(spec.dx, spec.dz) / (float(np.max(v)) * np.sqrt(2.0)):.3e} s')

    nz, nx = spec.shape
    area = spec.dx * spec.dz
    damp = abc.damping(spec) if abc is not None else np.ones_like(v)

    src_ids, src_w = spec.bilinear_weights(*src_xyz)
    recs = [spec.bilinear_weights(float(rx), float(rz)) for rx, rz in np.asarray(rec_xyz, float)]
    nr = len(recs)

    p = np.zeros(spec.shape)
    p_old = np.zeros_like(p)
    data = np.zeros((nt, nr))
    snaps = np.zeros((nt, nz, nx)) if store_wavefield else None

    for it in range(nt):
        _inject(p, src_ids, src_w, source[it], area)
        data[it] = [_sample(p, ids, w) for ids, w in recs]
        if store_wavefield:
            snaps[it] = p
        lap = _laplacian(p, spec.dx, spec.dz)
        p_new = 2.0 * p - p_old + (dt_s ** 2) * (v ** 2) * lap
        p_new *= damp
        p_old, p = p, p_new
        if it % 64 == 0:
            assert_finite(f'forward wavefield at time step {it}', p)
    assert_finite('forward wavefield (final)', p)
    assert_finite('synthetic shot gather', data)
    return ForwardResult(data=data, snapshots=snaps, dt_s=dt_s)
