import json
import numpy as np
import pytest
from geowave.fwi.config import FWIConfig, FrequencyBand
from geowave.fwi.demo import background_model, demo_config, true_model
from geowave.fwi.errors import FWIAbortError
from geowave.fwi.inversion import MultiscaleFWI
from geowave.fwi.linesearch import armijo_backtracking
from geowave.fwi.modeling import AcousticModeler2D, causal_ricker

def tiny_config(**kw):
    return demo_config(nx=40, nz=24, dx_m=10.0, tmax_s=0.3, n_shots=3, sponge=6,
                       bands=(FrequencyBand(4.0, 1), FrequencyBand(8.0, 1)), **kw)

def test_adjoint_gradient_matches_finite_difference(tmp_path):
    cfg = demo_config(nx=50, nz=30, tmax_s=0.35, n_shots=1, sponge=8)
    modeler = AcousticModeler2D(cfg.nz, cfg.nx, cfg.dx_m, cfg.dt, cfg.sponge_cells)
    rng = np.random.default_rng(7)
    v = np.clip(1600 + 250 * rng.standard_normal((cfg.nz, cfg.nx)), 1200, 2400)
    v_true = np.clip(v + 120 * rng.standard_normal((cfg.nz, cfg.nx)), 1200, 2400)
    fwi = MultiscaleFWI(cfg, tmp_path, modeler=modeler)
    wavelet = causal_ricker(8.0, cfg.dt)
    observed = fwi.generate_synthetic_shots(v_true, wavelet)  # data from a *different* model
    _, grad = fwi.objective_and_gradient(v, wavelet, observed, 'gradtest')
    iz = rng.integers(cfg.sponge_cells + 2, cfg.nz - cfg.sponge_cells - 1, 6)
    ix = rng.integers(cfg.sponge_cells + 1, cfg.nx - cfg.sponge_cells - 1, 6)
    eps = 1.0
    for z, x in zip(iz, ix):
        plus = v.copy(); plus[z, x] += eps
        minus = v.copy(); minus[z, x] -= eps
        fd = (fwi.objective(plus, wavelet, observed, 'fd+')
              - fwi.objective(minus, wavelet, observed, 'fd-')) / (2 * eps)
        assert abs(grad[z, x] - fd) / max(abs(fd), 1e-30) < 1e-2

def test_inversion_reduces_objective_and_snapshots(tmp_path):
    cfg = tiny_config()
    fwi = MultiscaleFWI(cfg, tmp_path)
    truth = true_model(cfg.nz, cfg.nx, cfg.dx_m)
    wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)
    observed = fwi.generate_synthetic_shots(truth, wavelet)
    result = fwi.run(background_model(cfg.nz, cfg.nx, cfg.dx_m), observed)
    assert result.status == 'completed'
    assert result.objectives[-1] < result.objectives[0]
    snaps = list((tmp_path / 'snapshots').glob('band*_model.npy'))
    assert snaps, 'expected per-iteration model snapshots'
    rows = (tmp_path / 'objective_log.csv').read_text().strip().splitlines()
    assert len(rows) == 1 + len(result.objectives)
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed'

def test_nonfinite_observed_aborts(tmp_path):
    cfg = tiny_config()
    fwi = MultiscaleFWI(cfg, tmp_path)
    observed = np.zeros((len(cfg.shot_x_m), cfg.nt, len(cfg.receiver_x_m)))
    observed[0, 10, 3] = np.nan
    result = fwi.run(background_model(cfg.nz, cfg.nx, cfg.dx_m), observed)
    assert result.status == 'aborted'
    assert 'observed data' in result.message and 'non-finite' in result.message
    assert (tmp_path / 'snapshots' / 'model_aborted.npy').exists()

class _NanForward(AcousticModeler2D):
    def forward(self, *a, **k):
        data, wf = super().forward(*a, **k)
        data[5, 0] = np.nan
        return data, wf

class _NanGradient(AcousticModeler2D):
    def adjoint_gradient(self, *a, **k):
        g = super().adjoint_gradient(*a, **k)
        g[8, 8] = np.inf
        return g

def _observed_with(cfg, tmp_path, modeler=None):
    fwi = MultiscaleFWI(cfg, tmp_path, modeler=modeler)
    truth = true_model(cfg.nz, cfg.nx, cfg.dx_m)
    wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)
    healthy = MultiscaleFWI(cfg, tmp_path / 'healthy')
    return fwi, healthy.generate_synthetic_shots(truth, wavelet)

def test_nonfinite_synthetic_aborts(tmp_path):
    cfg = tiny_config()
    fwi, observed = _observed_with(cfg, tmp_path, modeler=_NanForward(
        cfg.nz, cfg.nx, cfg.dx_m, cfg.dt, cfg.sponge_cells))
    result = fwi.run(background_model(cfg.nz, cfg.nx, cfg.dx_m), observed)
    assert result.status == 'aborted'
    assert 'synthetic shot' in result.message and 'non-finite' in result.message

def test_nonfinite_gradient_aborts(tmp_path):
    cfg = tiny_config()
    fwi, observed = _observed_with(cfg, tmp_path, modeler=_NanGradient(
        cfg.nz, cfg.nx, cfg.dx_m, cfg.dt, cfg.sponge_cells))
    result = fwi.run(background_model(cfg.nz, cfg.nx, cfg.dx_m), observed)
    assert result.status == 'aborted'
    assert 'gradient' in result.message and 'non-finite' in result.message

def test_linesearch_rollback_keeps_model():
    model = np.full((4, 4), 1500.0)
    gradient = np.full((4, 4), -2.0)     # objective increases above 1501
    direction = -gradient                # descent direction, but huge steps overshoot
    evaluate = lambda m: float(np.sum((m - 1501.0) ** 2))
    out = armijo_backtracking(evaluate, model, gradient, direction,
                              evaluate(model), initial_step=1e6,
                              vmin=700, vmax=2500, max_backtracks=2)
    assert not out.accepted
    assert np.array_equal(out.model, model)  # rolled back, untouched

def test_failed_linesearch_rolls_back_and_continues(tmp_path):
    cfg = tiny_config(initial_step_fraction=1.0, max_backtracks=1)
    fwi = MultiscaleFWI(cfg, tmp_path)
    truth = true_model(cfg.nz, cfg.nx, cfg.dx_m)
    wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)
    observed = fwi.generate_synthetic_shots(truth, wavelet)
    start = background_model(cfg.nz, cfg.nx, cfg.dx_m)
    result = fwi.run(start, observed)
    assert result.status == 'completed'
    assert np.allclose(result.final_model, np.clip(start, cfg.vmin_mps, cfg.vmax_mps))
    notes = (tmp_path / 'objective_log.csv').read_text()
    assert 'rollback' in notes

def test_box_constraint_respected(tmp_path):
    cfg = tiny_config()
    fwi = MultiscaleFWI(cfg, tmp_path)
    truth = true_model(cfg.nz, cfg.nx, cfg.dx_m)
    wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)
    observed = fwi.generate_synthetic_shots(truth, wavelet)
    result = fwi.run(background_model(cfg.nz, cfg.nx, cfg.dx_m), observed)
    assert result.final_model.min() >= cfg.vmin_mps
    assert result.final_model.max() <= cfg.vmax_mps
