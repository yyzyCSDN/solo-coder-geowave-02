from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from geowave.inversion.fwi.adjoint import ObjectiveGradient, objective_and_gradient
from geowave.inversion.fwi.archive import RunArchive
from geowave.inversion.fwi.finite import FWINonFiniteError, assert_finite
from geowave.inversion.fwi.filters import FrequencyBand
from geowave.inversion.fwi.grid import GridSpec, clamp_velocity
from geowave.inversion.fwi.linesearch import backtracking_line_search
from geowave.inversion.fwi.precond import descent_direction
from geowave.inversion.fwi.survey import Survey
from geowave.inversion.fwi.wave import AbsorbingBoundary
from geowave.model.wavelet import Wavelet


@dataclass(frozen=True)
class FWIConfig:
    max_iterations_per_band: int = 10
    vmin_mps: float = 1400.0
    vmax_mps: float = 4000.0
    initial_step_mps: float = 120.0
    step_shrink: float = 0.5
    max_line_search_trials: int = 10
    armijo_c1: float = 1e-4
    smooth_radius_m: float = 0.0
    objective_tolerance: float = 1e-5


@dataclass
class FWIResult:
    velocity: np.ndarray
    archive: RunArchive
    history: list[dict] = field(default_factory=list)
    status: str = 'completed'
    reason: str = ''


def run_fwi(initial_velocity: np.ndarray, spec: GridSpec, observed: np.ndarray,
            dt_s: float, wavelet: Wavelet, survey: Survey,
            bands: tuple[FrequencyBand, ...], abc: AbsorbingBoundary | None,
            config: FWIConfig, archive_root: str | Path) -> FWIResult:
    """Multiscale (low->high) FWI with velocity bounds, rollback line search.

    Every iteration appends an objective record to history.jsonl and stores a
    model snapshot. Non-finite data, wavefields or gradients abort the run with
    FWINonFiniteError; the manifest records the failing stage and the best
    model is left on disk.
    """
    v = np.array(initial_velocity, dtype=float)
    # Structural / positivity guards on the configuration itself.
    assert_finite('initial velocity model', v)
    assert_finite('velocity bounds', np.array([config.vmin_mps, config.vmax_mps]))
    if np.any(v <= 0):
        raise ValueError('initial velocity must be positive')
    if config.vmin_mps >= config.vmax_mps:
        raise ValueError('vmin must be below vmax')

    archive = RunArchive(archive_root, config={
        'grid': {'nx': spec.nx, 'nz': spec.nz, 'dx': spec.dx, 'dz': spec.dz},
        'dt_s': dt_s,
        'nt': int(observed.shape[1]),
        'n_shots': len(survey),
        'wavelet': wavelet.name,
        'bands': [str(b) for b in bands],
        'vmin_mps': config.vmin_mps,
        'vmax_mps': config.vmax_mps,
    })
    archive.save_snapshot(0, 'initial', v)
    result = FWIResult(velocity=v, archive=archive)
    iteration = 0
    try:
        # Data guard lives inside the run so a bad dataset aborts (and records)
        # exactly like a non-finite wavefield or gradient would.
        assert_finite('observed shot gathers', observed)
        for band in bands:
            for _ in range(config.max_iterations_per_band):
                iteration += 1
                og: ObjectiveGradient = objective_and_gradient(
                    v, spec, observed, dt_s, wavelet.samples, wavelet.dt_s,
                    tuple(survey.shots), band, abc)
                assert_finite(f'objective for {band}', np.asarray(og.objective))
                assert_finite(f'FWI gradient for {band}', og.gradient)
                direction = descent_direction(og.gradient, spec,
                                               config.smooth_radius_m)

                def evaluate(candidate: np.ndarray) -> float:
                    trial = objective_and_gradient(
                        candidate, spec, observed, dt_s, wavelet.samples,
                        wavelet.dt_s, tuple(survey.shots), band, abc)
                    return trial.objective

                ls = backtracking_line_search(
                    v, direction, og.objective, og.gradient, evaluate,
                    vmin=config.vmin_mps, vmax=config.vmax_mps,
                    initial_step=config.initial_step_mps,
                    shrink=config.step_shrink,
                    max_trials=config.max_line_search_trials,
                    c1=config.armijo_c1)
                record = {
                    'iteration': iteration,
                    'band': str(band),
                    'objective_before': og.objective,
                    'objective_after': ls.objective if ls.accepted else og.objective,
                    'per_shot_objective': list(og.per_shot_objective),
                    'grad_norm': float(np.linalg.norm(og.gradient)),
                    'step_mps': ls.step,
                    'line_search': [e for e in ls.evaluations],
                    'accepted': ls.accepted,
                    'reason': ls.reason,
                    'vmin_mps': float(np.min(v)),
                    'vmax_mps': float(np.max(v)),
                }
                if ls.accepted:
                    v = clamp_velocity(ls.model, config.vmin_mps, config.vmax_mps)
                    record['vmin_mps'] = float(np.min(v))
                    record['vmax_mps'] = float(np.max(v))
                archive.save_snapshot(iteration, str(band), v)
                archive.append_history(record)
                result.history.append(record)
                result.velocity = v
                if (ls.accepted
                        and abs(og.objective - ls.objective) <=
                        config.objective_tolerance * max(abs(og.objective), 1e-30)):
                    break
        archive.finalize('completed')
        result.status = 'completed'
    except FWINonFiniteError as exc:
        reason = exc.detail.describe()
        # Preserve the last finite model snapshot before surfacing the abort.
        archive.save_snapshot(iteration if iteration else 0, 'abort', v)
        archive.append_history({'iteration': iteration, 'event': 'abort',
                                'reason': reason})
        archive.finalize('aborted', reason=reason)
        result.status = 'aborted'
        result.reason = reason
        exc.result = result
        raise
    return result
