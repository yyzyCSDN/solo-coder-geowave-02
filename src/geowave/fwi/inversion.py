from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from geowave.fwi.config import FWIConfig
from geowave.fwi.errors import FWIAbortError, require_finite
from geowave.fwi.filters import lowpass_zero_phase
from geowave.fwi.history import IterationRecord, RunHistory, write_manifest
from geowave.fwi.linesearch import armijo_backtracking
from geowave.fwi.modeling import AcousticModeler2D, causal_ricker

@dataclass(frozen=True)
class FWIRunResult:
    status: str            # 'completed' | 'aborted'
    message: str
    final_model: np.ndarray
    objectives: tuple[float, ...]
    workdir: Path
    records: tuple = ()    # IterationRecord per gradient evaluation

class MultiscaleFWI:
    """Multi-scale (low-to-high frequency) acoustic FWI with box constraints,
    Armijo line search with rollback, per-iteration model snapshots and
    abort-on-nonfinite data/gradient policy."""

    def __init__(self, config: FWIConfig, workdir: str | Path, modeler: AcousticModeler2D | None=None):
        self.config = config
        self.workdir = Path(workdir)
        self.modeler = modeler or AcousticModeler2D(
            config.nz, config.nx, config.dx_m, config.dt,
            sponge_cells=config.sponge_cells, free_surface=config.free_surface)
        self.receivers = (np.full(len(config.receiver_x_m), round(config.receiver_z_m / config.dx_m), dtype=int),
                          np.round(np.asarray(config.receiver_x_m) / config.dx_m).astype(int))
        self.shots = [(round(config.source_z_m / config.dx_m), round(x / config.dx_m))
                      for x in config.shot_x_m]
        self.snapshots = self.workdir / 'snapshots'
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self._illumination: np.ndarray | None = None

    # ------------------------------------------------------------------ data
    def source_function(self, wavelet: np.ndarray) -> np.ndarray:
        """Wavelet zero-padded/truncated to the modeling time axis (cfg.nt)."""
        w = np.zeros(self.config.nt)
        n = min(wavelet.size, self.config.nt)
        w[:n] = wavelet[:n]
        return w

    def generate_synthetic_shots(self, model: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """Synthetic shot gather for every shot: returns (nshots, nt, nrec)."""
        require_finite(model, 'model', 'generate_synthetic_shots')
        wavelet = self.source_function(wavelet)
        gathers = []
        for ishot, src in enumerate(self.shots):
            data, _ = self.modeler.forward(model, wavelet, src, self.receivers)
            require_finite(data, f'synthetic shot {ishot}', 'generate_synthetic_shots')
            gathers.append(data)
        return np.stack(gathers)

    def save_observed(self, observed: np.ndarray) -> Path:
        path = self.workdir / 'observed.npy'
        np.save(path, observed)
        return path

    # ------------------------------------------------------- objective/gradient
    def objective(self, model: np.ndarray, wavelet: np.ndarray, observed: np.ndarray, context: str) -> float:
        wavelet = self.source_function(wavelet)
        total = 0.0
        for ishot, src in enumerate(self.shots):
            data, _ = self.modeler.forward(model, wavelet, src, self.receivers)
            require_finite(data, f'synthetic shot {ishot}', context)
            residual = data - observed[ishot]
            total += 0.5 * self.config.dt * float(np.sum(residual ** 2))
        return total

    def objective_and_gradient(self, model: np.ndarray, wavelet: np.ndarray,
                               observed: np.ndarray, context: str) -> tuple[float, np.ndarray]:
        wavelet = self.source_function(wavelet)
        total = 0.0
        gradient = np.zeros_like(model, dtype=float)
        illumination = np.zeros_like(model, dtype=float)
        for ishot, src in enumerate(self.shots):
            data, wavefield = self.modeler.forward(model, wavelet, src, self.receivers, record_wavefield=True)
            require_finite(data, f'synthetic shot {ishot}', context)
            residual = data - observed[ishot]
            total += 0.5 * self.config.dt * float(np.sum(residual ** 2))
            illumination += self.config.dt * np.sum(wavefield ** 2, axis=0)
            shot_gradient = self.modeler.adjoint_gradient(model, residual, self.receivers, wavefield)
            require_finite(shot_gradient, f'adjoint gradient shot {ishot}', context)
            gradient += shot_gradient
        require_finite(gradient, 'stacked gradient', context)
        self._illumination = illumination
        return total, gradient

    # ------------------------------------------------------------------- run
    def run(self, initial_model: np.ndarray, observed: np.ndarray) -> FWIRunResult:
        cfg = self.config
        history = RunHistory(self.workdir)
        objectives: list[float] = []
        status, message = 'completed', 'all bands finished'
        model = np.asarray(initial_model, float).copy()
        try:
            require_finite(model, 'initial model', 'run start')
            require_finite(observed, 'observed data', 'run start')
            expected = (len(self.shots), cfg.nt, self.receivers[0].size)
            if observed.shape != expected:
                raise FWIAbortError(f'observed data shape {observed.shape} != expected {expected}')
            model = np.clip(model, cfg.vmin_mps, cfg.vmax_mps)
            if not cfg.cfl_ok():
                raise FWIAbortError(f'dt={cfg.dt:.3e}s violates the CFL limit at vmax={cfg.vmax_mps:g}')
            if cfg.points_per_wavelength() < 5.0:
                history.message(f'WARNING: only {cfg.points_per_wavelength():.1f} points per shortest '
                                f'wavelength; expect numerical dispersion')
            (self.workdir / 'config.json').write_text(cfg.to_json())
            np.save(self.snapshots / 'model_init.npy', model)
            base_wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)

            for iband, band in enumerate(cfg.bands):
                wavelet = lowpass_zero_phase(base_wavelet, cfg.dt, band.lowpass_hz)
                band_observed = lowpass_zero_phase(observed, cfg.dt, band.lowpass_hz, axis=1)
                require_finite(band_observed, 'filtered observed data', f'band {iband}')
                step = cfg.initial_step_fraction * (cfg.vmax_mps - cfg.vmin_mps)
                history.message(f'--- band {iband}: lowpass {band.lowpass_hz:g} Hz, '
                                f'{band.max_iterations} iterations ---')
                for iteration in range(band.max_iterations):
                    context = f'band {iband} ({band.lowpass_hz:g} Hz) iteration {iteration}'
                    objective, gradient = self.objective_and_gradient(model, wavelet, band_observed, context)
                    objectives.append(objective)
                    grad_max = float(np.max(np.abs(gradient)))
                    if grad_max == 0.0:
                        message = f'zero gradient at {context}; converged'
                        history.message(message)
                        break
                    direction = -self._precondition(gradient)
                    direction /= np.max(np.abs(direction))

                    def trial_logger(trial, step_, obj_, accepted_):
                        history.log_trial({'band': iband, 'iteration': iteration, 'trial': trial,
                                           'step': step_, 'objective': obj_, 'accepted': accepted_})

                    outcome = armijo_backtracking(
                        lambda m: self.objective(m, wavelet, band_observed, context + ' (line search)'),
                        model, gradient, direction, objective, step,
                        cfg.vmin_mps, cfg.vmax_mps,
                        c=cfg.armijo_c, shrink=cfg.step_shrink,
                        max_backtracks=cfg.max_backtracks, on_trial=trial_logger)
                    if outcome.accepted:
                        model = outcome.model
                        name = f'band{iband:02d}_iter{iteration:03d}_model.npy'
                        np.save(self.snapshots / name, model)
                        if cfg.save_gradient_snapshots:
                            np.save(self.snapshots / name.replace('model', 'gradient'), gradient)
                        np.save(self.snapshots / 'model_latest.npy', model)
                        step = min(outcome.step / cfg.step_shrink,
                                   0.5 * (cfg.vmax_mps - cfg.vmin_mps))
                        note = outcome.message
                    else:
                        # rollback: keep the previous model, move to the next band
                        name = 'model_latest.npy (unchanged)'
                        note = f'rollback: {outcome.message}'
                        history.message(f'line search failed at {context}; model rolled back')
                    history.log_record(IterationRecord(
                        iband, band.lowpass_hz, iteration, objective, outcome.step,
                        grad_max, float(model.min()), float(model.max()),
                        outcome.accepted, name, note))
                    if not outcome.accepted:
                        break
        except FWIAbortError as err:
            status, message = 'aborted', str(err)
            history.message(f'ABORT: {message}')
            np.save(self.snapshots / 'model_aborted.npy', model)
        finally:
            np.save(self.snapshots / 'model_final.npy', model)
            write_manifest(self.workdir, {
                'status': status, 'message': message, 'config': cfg.to_json(),
                'objectives': objectives,
                'snapshots': sorted(p.name for p in self.snapshots.glob('*.npy'))})
            history.message(f'run finished: {status} — {message}')
            history.close()
        return FWIRunResult(status, message, model, tuple(objectives), self.workdir,
                            tuple(history.records))

    def _precondition(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, float)
        if self.config.illumination_preconditioner and self._illumination is not None:
            illum = self._illumination
            g = g / (illum + self.config.illumination_epsilon * illum.max())
        g = np.where(self.modeler.gradient_mask(), g, 0.0)
        for _ in range(self.config.gradient_smoothing_cells):
            p = np.pad(g, 1, mode='edge')
            g = sum(p[di:di + g.shape[0], dj:dj + g.shape[1]]
                    for di in range(3) for dj in range(3)) / 9.0
        return g
