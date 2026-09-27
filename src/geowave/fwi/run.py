from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from geowave.fwi.config import FWIConfig
from geowave.fwi.demo import background_model, demo_config, true_model
from geowave.fwi.inversion import MultiscaleFWI
from geowave.fwi.modeling import AcousticModeler2D, causal_ricker

def gradient_test() -> int:
    """Adjoint gradient vs centred finite differences of the misfit."""
    from geowave.fwi.demo import demo_config
    cfg = demo_config(nx=50, nz=30, tmax_s=0.35, n_shots=1, sponge=8)
    modeler = AcousticModeler2D(cfg.nz, cfg.nx, cfg.dx_m, cfg.dt, cfg.sponge_cells)
    rng = np.random.default_rng(7)
    v = np.clip(1600 + 250 * rng.standard_normal((cfg.nz, cfg.nx)), 1200, 2400)
    v_true = np.clip(v + 120 * rng.standard_normal((cfg.nz, cfg.nx)), 1200, 2400)
    fwi = MultiscaleFWI(cfg, Path('/tmp/fwi_gradtest'), modeler=modeler)
    wavelet = causal_ricker(8.0, cfg.dt)
    observed = fwi.generate_synthetic_shots(v_true, wavelet)  # data from a *different* model
    obj, grad = fwi.objective_and_gradient(v, wavelet, observed, 'gradient test')
    # perturbation points kept clear of the sponge frame
    iz = rng.integers(cfg.sponge_cells + 2, cfg.nz - cfg.sponge_cells - 1, 6)
    ix = rng.integers(cfg.sponge_cells + 1, cfg.nx - cfg.sponge_cells - 1, 6)
    eps = 1.0
    rel_errors = []
    print(f'{"point":>12} {"adjoint":>14} {"finite-diff":>14} {"rel.err":>10}')
    for z, x in zip(iz, ix):
        plus = v.copy(); plus[z, x] += eps
        minus = v.copy(); minus[z, x] -= eps
        fd = (fwi.objective(plus, wavelet, observed, 'fd+')
              - fwi.objective(minus, wavelet, observed, 'fd-')) / (2 * eps)
        rel = abs(grad[z, x] - fd) / max(abs(fd), 1e-30)
        rel_errors.append(rel)
        print(f'({z:3d},{x:3d})   {grad[z, x]:14.6e} {fd:14.6e} {rel:10.2e}')
    worst = max(rel_errors)
    print(f'worst relative error: {worst:.3e} (tolerance 1e-2)')
    return 0 if worst < 1e-2 else 1

def _load_production(args) -> tuple[np.ndarray, np.ndarray]:
    if not args.init_model:
        raise SystemExit('--obs-dir requires --init-model (e.g. the ray-tomography model)')
    obs_path = Path(args.obs_dir)
    shots = sorted(obs_path.glob('shot_*.npy'))
    if not shots:
        raise SystemExit(f'no shot_*.npy files in {obs_path}')
    observed = np.stack([np.load(p) for p in shots])
    initial = np.load(args.init_model)
    return observed, initial

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser(prog='geowave-fwi', description='Multi-scale full-waveform inversion')
    p.add_argument('--workdir', default='runs/fwi')
    p.add_argument('--gradient-test', action='store_true', help='verify adjoint gradient and exit')
    p.add_argument('--self-test', action='store_true', help='generate observed data from the demo true model')
    p.add_argument('--obs-dir', help='directory with observed shot_*.npy (nt, nrec) files')
    p.add_argument('--init-model', help='npy initial model, e.g. the ray-tomography result')
    p.add_argument('--nx', type=int, default=121)
    p.add_argument('--nz', type=int, default=61)
    p.add_argument('--dx', type=float, default=10.0)
    p.add_argument('--tmax', type=float, default=1.0)
    p.add_argument('--shots', type=int, default=7)
    p.add_argument('--bands', default='2.5:5,5:5,10:5', help='cutoff_hz:iterations,...')
    p.add_argument('--vmin', type=float, default=700.0)
    p.add_argument('--vmax', type=float, default=2500.0)
    p.add_argument('--wavelet-peak', type=float, default=6.0)
    p.add_argument('--step-fraction', type=float, default=0.05)
    p.add_argument('--max-backtracks', type=int, default=8)
    args = p.parse_args(argv)

    if args.gradient_test:
        return gradient_test()

    cfg = demo_config(nx=args.nx, nz=args.nz, dx_m=args.dx, tmax_s=args.tmax,
                      n_shots=args.shots, bands=FWIConfig.parse_bands(args.bands),
                      vmin_mps=args.vmin, vmax_mps=args.vmax,
                      wavelet_peak_hz=args.wavelet_peak,
                      initial_step_fraction=args.step_fraction,
                      max_backtracks=args.max_backtracks)
    fwi = MultiscaleFWI(cfg, args.workdir)
    if args.obs_dir:
        observed, initial = _load_production(args)
    else:
        truth = true_model(cfg.nz, cfg.nx, cfg.dx_m)
        np.save(Path(args.workdir) / 'true_model.npy', truth)
        wavelet = causal_ricker(cfg.wavelet_peak_hz, cfg.dt)
        observed = fwi.generate_synthetic_shots(truth, wavelet)
        fwi.save_observed(observed)
        initial = (np.load(args.init_model) if args.init_model
                   else background_model(cfg.nz, cfg.nx, cfg.dx_m))
    np.save(Path(args.workdir) / 'initial_model.npy', initial)
    result = fwi.run(initial, observed)
    print(f'status: {result.status} — {result.message}')
    for iband, band in enumerate(cfg.bands):
        objs = [r.objective for r in result.records if r.band_index == iband]
        if objs:
            print(f'band {iband} ({band.lowpass_hz:g} Hz): objective {objs[0]:.6e} -> {objs[-1]:.6e}')
    return 0 if result.status == 'completed' else 3

if __name__ == '__main__':
    sys.exit(main())
