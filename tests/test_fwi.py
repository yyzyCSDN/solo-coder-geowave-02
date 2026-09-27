import json
import numpy as np
import pytest

from geowave.inversion.fwi import (
    AbsorbingBoundary, FWINonFiniteError, FrequencyBand, FWIConfig, GridSpec,
    RunArchive, continuation_bands, generate_synthetic, run_fwi, stable_dt,
    surface_survey,
)
from geowave.inversion.fwi.adjoint import objective_and_gradient
from geowave.inversion.fwi.filters import bandpass
from geowave.inversion.fwi.grid import clamp_velocity
from geowave.inversion.fwi.linesearch import backtracking_line_search
from geowave.model.wavelet import ricker


@pytest.fixture(scope='module')
def setup():
    dx = dz = 10.0
    x = np.arange(0, 500 + dx, dx)
    z = np.arange(0, 300 + dz, dz)
    spec = GridSpec(x, z)
    gx, gz = np.meshgrid(x, z)
    zz = z[:, None]
    v_bg = 1800.0 + 800.0 * (zz / z[-1])
    body = 250.0 * np.exp(-(((gx - 250) / 120.0) ** 2 + ((zz - 50) / 30.0) ** 2))
    v_true = v_bg - body
    survey = surface_survey(np.arange(60.0, 480.0, 140.0),
                            np.arange(30.0, 480.0, 20.0), dz, dz)
    wavelet = ricker(18.0, 0.002, length_s=0.128)
    abc = AbsorbingBoundary(width=12, strength=0.02, damp_top=False)
    data = generate_synthetic(v_true, spec, survey, wavelet, 0.4, abc)
    v_init = np.broadcast_to(v_bg, spec.shape).copy()
    return dict(spec=spec, v_true=v_true, v_init=v_init, survey=survey,
                wavelet=wavelet, abc=abc, data=data)


def test_synthetic_gathers_finite_and_shaped(setup):
    d = setup['data']
    nshots = len(setup['survey'])
    nr = setup['survey'].shots[0].n_receivers
    assert d.gathers.shape == (nshots, d.gathers.shape[1], nr)
    assert np.isfinite(d.gathers).all()
    assert d.gathers.std() > 0
    assert d.dt_s <= stable_dt(setup['spec'], float(np.max(setup['v_true'])))


def test_stable_dt_respects_cfl():
    spec = GridSpec(np.arange(0, 100, 5.0), np.arange(0, 100, 5.0))
    dt = stable_dt(spec, 3000.0, cfl=0.45)
    assert dt == pytest.approx(0.45 * 5.0 / (3000.0 * np.sqrt(2.0)))


def test_adjoint_gradient_dot_product(setup):
    s = setup
    band = FrequencyBand(0.0, 26.0)
    og = objective_and_gradient(s['v_init'], s['spec'], s['data'].gathers, s['data'].dt_s,
                                s['wavelet'].samples, s['wavelet'].dt_s,
                                s['survey'].shots, band, s['abc'])
    assert np.isfinite(og.gradient).all()
    rng = np.random.default_rng(3)
    dm = 0.05 * rng.standard_normal(s['spec'].shape)
    analytic = float(np.sum(og.gradient * dm))
    eps = 0.5
    ogp = objective_and_gradient(s['v_init'] + eps * dm, s['spec'], s['data'].gathers,
                                 s['data'].dt_s, s['wavelet'].samples,
                                 s['wavelet'].dt_s, s['survey'].shots, band, s['abc'])
    finite_diff = (ogp.objective - og.objective) / eps
    assert finite_diff == pytest.approx(analytic, rel=0.05)


def test_frequency_continuation_order_and_filter():
    bands = continuation_bands(8.0, 32.0, factor=2.0)
    highs = [b.f_high_hz for b in bands]
    assert highs == sorted(highs) and highs[0] == 8.0 and highs[-1] == 32.0
    t = np.arange(2048) * 0.002
    sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 60 * t)
    low = bandpass(sig, 0.002, FrequencyBand(0.0, 15.0))
    # 60 Hz removed, ~5 Hz retained (edge effects ignored in the middle)
    seg = low[200:-200]
    assert seg.std() > 0.1
    assert np.ptp(low[200:-200]) < 2.2


def test_clamp_enforces_velocity_bounds():
    v = np.array([[1200.0, 2000.0], [5000.0, 3000.0]])
    c = clamp_velocity(v, 1400.0, 4000.0)
    assert c.min() >= 1400.0 and c.max() <= 4000.0
    with pytest.raises(ValueError):
        clamp_velocity(v, 4000.0, 1400.0)


def test_line_search_rolls_back_on_nonfinite_objective():
    m0 = np.array([2000.0, 2000.0])
    d = np.array([-1.0, -1.0])
    g = np.array([1.0, 1.0])

    def evaluate(candidate):
        if candidate[0] < 1950.0:
            return np.nan
        return 0.5  # small step stays near 2000 and is fine

    res = backtracking_line_search(m0, d, 1.0, g, evaluate,
                                   initial_step=100.0, shrink=0.5, max_trials=12)
    assert res.accepted
    assert np.all(np.isfinite(res.model))
    assert res.step < 100.0  # large (non-finite) steps were rolled back and shrunk
    # a non-finite large trial was recorded as rejected, not raised
    assert any('non-finite objective' in (e.get('rejected') or '')
               for e in res.evaluations)


def test_line_search_failure_leaves_model_unchanged():
    m0 = np.array([2000.0, 2000.0])
    res = backtracking_line_search(m0, np.array([-1.0, 0.0]), 1.0,
                                   np.array([1.0, 0.0]), lambda c: 2.0,
                                   initial_step=10.0, max_trials=3)
    assert not res.accepted
    assert np.array_equal(res.model, m0)
    assert 'no acceptable step' in res.reason


def test_nonfinite_observed_data_aborts_with_reason(setup, tmp_path):
    s = setup
    bad = s['data'].gathers.copy()
    bad[0, 10, 3] = np.inf
    cfg = FWIConfig(max_iterations_per_band=1, initial_step_mps=20.0)
    with pytest.raises(FWINonFiniteError) as exc:
        run_fwi(s['v_init'], s['spec'], bad, s['data'].dt_s, s['wavelet'],
                s['survey'], (FrequencyBand(0.0, 15.0),), s['abc'], cfg, tmp_path)
    assert 'observed' in str(exc.value)
    result = exc.value.result
    assert result.status == 'aborted'
    assert result.reason
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['status'] == 'aborted'
    assert 'infinite' in manifest['reason']
    # history records the abort event and the last finite snapshot is on disk
    hist = [json.loads(l) for l in (tmp_path / 'history.jsonl').read_text().splitlines()]
    assert hist[-1]['event'] == 'abort'
    assert (tmp_path / 'snapshots' / 'iter_0000.npz').exists()


def test_nonfinite_gradient_aborts(setup, tmp_path, monkeypatch):
    s = setup
    import geowave.inversion.fwi.run as run_mod

    real = run_mod.objective_and_gradient
    calls = {'n': 0}

    def poisoned(*args, **kwargs):
        out = real(*args, **kwargs)
        calls['n'] += 1
        if calls['n'] == 1:
            out.gradient[0, 0] = np.nan  # bypass the in-function guard to test run guard path
        return out

    monkeypatch.setattr(run_mod, 'objective_and_gradient', poisoned)
    cfg = FWIConfig(max_iterations_per_band=1, initial_step_mps=20.0)
    from geowave.inversion.fwi.finite import FWINonFiniteError as E
    with pytest.raises(E):
        run_fwi(s['v_init'], s['spec'], s['data'].gathers, s['data'].dt_s, s['wavelet'],
                s['survey'], (FrequencyBand(0.0, 15.0),), s['abc'], cfg, tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['status'] == 'aborted'


def test_full_run_records_history_snapshots_and_improves(setup, tmp_path):
    s = setup
    bands = continuation_bands(10.0, 24.0, factor=2.0)
    cfg = FWIConfig(max_iterations_per_band=3, vmin_mps=1500.0, vmax_mps=2900.0,
                    initial_step_mps=40.0, smooth_radius_m=20.0)
    res = run_fwi(s['v_init'], s['spec'], s['data'].gathers, s['data'].dt_s,
                  s['wavelet'], s['survey'], bands, s['abc'], cfg, tmp_path)
    assert res.status == 'completed'

    archive = RunArchive(tmp_path)
    hist = archive.load_history()
    iterations = [h for h in hist if 'objective_before' in h]
    assert len(iterations) == 3 * len(bands)
    # objectives are monotonically non-increasing within the run
    objs = [h['objective_after'] for h in iterations]
    assert all(b <= a + 1e-6 for a, b in zip(objs, objs[1:]))
    # velocity bounds respected by every snapshot
    for h in iterations:
        assert h['vmin_mps'] >= cfg.vmin_mps - 1e-9
        assert h['vmax_mps'] <= cfg.vmax_mps + 1e-9
    snaps = sorted((tmp_path / 'snapshots').glob('iter_*.npz'))
    assert len(snaps) == len(iterations) + 1  # initial + per-iteration
    # snapshot 0 is the starting model; last snapshot is the result
    with np.load(snaps[0]) as z:
        assert np.allclose(z['velocity'], s['v_init'])
    assert np.allclose(archive.latest_snapshot(), res.velocity)
    # fit improves in the anomaly region
    body = np.abs(s['v_true'] - s['v_init']) > 50
    e0 = np.linalg.norm(s['v_init'][body] - s['v_true'][body])
    ef = np.linalg.norm(res.velocity[body] - s['v_true'][body])
    assert ef < e0
