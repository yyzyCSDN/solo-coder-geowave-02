from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.inversion.fwi.finite import assert_finite
from geowave.inversion.fwi.filters import FrequencyBand, bandpass
from geowave.inversion.fwi.grid import GridSpec
from geowave.inversion.fwi.survey import ShotGeometry
from geowave.inversion.fwi.wave import AbsorbingBoundary, ForwardResult, _laplacian


@dataclass(frozen=True)
class ObjectiveGradient:
    objective: float
    gradient: np.ndarray          # dPhi/dv, averaged over shots
    per_shot_objective: tuple[float, ...]


def _source_trace(source_samples: np.ndarray, source_dt_s: float, dt_s: float, nt: int) -> np.ndarray:
    t0 = np.arange(source_samples.size) * source_dt_s
    t = np.arange(nt) * dt_s
    return np.interp(t, t0 - t0[0], source_samples, left=0.0, right=0.0)


def _adjoint_single(v2: np.ndarray, spec: GridSpec, dt_s: float, source: np.ndarray,
                    shot: ShotGeometry, observed_f: np.ndarray, abc: AbsorbingBoundary | None,
                    ) -> tuple[float, np.ndarray, np.ndarray]:
    """Forward + exact discrete adjoint for one shot.

    Returns (objective, gradient wrt m=v^2, predicted gather). The reverse
    sweep is line-by-line transpose of the forward recurrence
    ``u[t+1] = src + D (2 u[t] - u[t-1] + dt^2 M L u[t])`` with data
    ``d[t] = R u[t]``; hence R^T injects without the source cell-area factor
    and the spatial term is transposed as L(M .), not M(L .).
    """
    from geowave.inversion.fwi.wave import forward_model

    nt, nr = observed_f.shape
    fwd: ForwardResult = forward_model(v2 ** 0.5, spec, dt_s, nt, source,
                                       shot.source_xz, shot.receiver_xz, abc,
                                       store_wavefield=True)
    predicted = fwd.data
    assert_finite(f'predicted gather for {shot.shot_id}', predicted)
    residual = predicted - observed_f
    assert_finite(f'data residual for {shot.shot_id}', residual)

    damp = abc.damping(spec) if abc is not None else np.ones_like(v2)
    recs = [spec.bilinear_weights(float(rx), float(rz)) for rx, rz in shot.receiver_xz]

    def scatter_residual(values: np.ndarray) -> np.ndarray:
        out = np.zeros(spec.shape)
        for (ids, w), rj in zip(recs, values):   # R^T: no cell-area factor
            for (iz, ix), ww in zip(ids, w):
                out[iz, ix] += rj * ww
        return out

    g_m = np.zeros(spec.shape)           # gradient wrt m = v^2
    bar_next = np.zeros(spec.shape)      # bar[k+1]
    bar_next2 = np.zeros(spec.shape)     # bar[k+2]
    for k in range(nt - 1, 0, -1):
        bar = scatter_residual(residual[k])
        bar += damp * (2.0 * bar_next - bar_next2
                       + (dt_s ** 2) * _laplacian(v2 * bar_next, spec.dx, spec.dz))
        w = damp * bar
        g_m += (dt_s ** 2) * _laplacian(fwd.snapshots[k - 1], spec.dx, spec.dz) * w
        bar_next2, bar_next = bar_next, bar
        if k % 64 == 0:
            assert_finite(f'adjoint wavefield for {shot.shot_id} at step {k}', bar)
    objective = 0.5 * float(np.sum(residual ** 2))
    assert_finite(f'adjoint gradient for {shot.shot_id}', g_m)
    return objective, g_m, predicted


def objective_and_gradient(velocity: np.ndarray, spec: GridSpec,
                           observed: np.ndarray, dt_s: float,
                           source_samples: np.ndarray, source_dt_s: float,
                           shots: tuple[ShotGeometry, ...],
                           band: FrequencyBand,
                           abc: AbsorbingBoundary | None = None
                           ) -> ObjectiveGradient:
    """L2 least-squares objective and adjoint gradient at one frequency band.

    observed : (nshots, nt, nr) field gathers; both data legs are filtered with
    the same zero-phase filter before differencing.
    """
    v = np.asarray(velocity, dtype=float)
    if v.shape != spec.shape:
        raise ValueError('velocity shape does not match grid')
    assert_finite('velocity model', v)
    if np.any(v <= 0):
        raise ValueError('velocity must be positive everywhere')
    nshots, nt, nr = observed.shape
    if len(shots) != nshots:
        raise ValueError('shot/observed count mismatch')
    source = _source_trace(np.asarray(source_samples, float), source_dt_s, dt_s, nt)
    assert_finite('source wavelet', source)

    v2 = v ** 2
    g_total = np.zeros(spec.shape)
    obj_shots: list[float] = []
    for k, shot in enumerate(shots):
        if shot.n_receivers != nr:
            raise ValueError(f'receiver count mismatch on {shot.shot_id}')
        obs_f = bandpass(observed[k], dt_s, band)
        assert_finite(f'filtered observed gather for {shot.shot_id}', obs_f)
        obj_k, g_m, _ = _adjoint_single(v2, spec, dt_s, source, shot, obs_f, abc)
        obj_shots.append(obj_k)
        g_total += g_m

    g_total /= nshots
    # chain rule: dPhi/dv = 2 v dPhi/d(v^2)
    gradient = 2.0 * v * g_total
    assert_finite('FWI gradient', gradient)
    return ObjectiveGradient(objective=float(sum(obj_shots) / nshots),
                             gradient=gradient,
                             per_shot_objective=tuple(obj_shots))
