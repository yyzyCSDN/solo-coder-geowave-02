from __future__ import annotations
import numpy as np
from geowave.fwi.config import FWIConfig, FrequencyBand

def true_model(nz: int, nx: int, dx_m: float) -> np.ndarray:
    """Demo near-surface model: vertical gradient + fast deep block + slow shallow lens."""
    z, x = np.meshgrid(np.arange(nz) * dx_m, np.arange(nx) * dx_m, indexing='ij')
    lz, lx = z.max(), x.max()
    v = 800.0 + 1500.0 * z / lz
    v = v + 320.0 * np.exp(-((x - 0.52 * lx) ** 2 / (2 * (0.08 * lx) ** 2)
                             + (z - 0.62 * lz) ** 2 / (2 * (0.10 * lz) ** 2)))
    v = v - 260.0 * np.exp(-((x - 0.30 * lx) ** 2 / (2 * (0.06 * lx) ** 2)
                             + (z - 0.20 * lz) ** 2 / (2 * (0.05 * lz) ** 2)))
    return np.clip(v, 700.0, 2500.0)

def background_model(nz: int, nx: int, dx_m: float) -> np.ndarray:
    """1D vertical gradient only — the demo starting model (e.g. from ray tomography)."""
    z = np.arange(nz) * dx_m
    v = 800.0 + 1500.0 * z / z.max()
    return np.repeat(v[:, None], nx, axis=1)

def demo_config(nx: int=121, nz: int=61, dx_m: float=10.0, tmax_s: float=1.0,
                n_shots: int=7, bands: tuple[FrequencyBand, ...] | None=None,
                sponge: int=15, **overrides) -> FWIConfig:
    margin = (sponge + 2) * dx_m
    xs = np.linspace(margin, (nx - 1) * dx_m - margin, n_shots)
    xr = np.arange(margin, (nx - 1) * dx_m - margin + 1, 2 * dx_m)
    params = dict(nx=nx, nz=nz, dx_m=dx_m, tmax_s=tmax_s,
                  shot_x_m=tuple(float(v) for v in xs),
                  receiver_x_m=tuple(float(v) for v in xr),
                  vmin_mps=700.0, vmax_mps=2500.0,
                  bands=bands or (FrequencyBand(2.5, 5), FrequencyBand(5.0, 5), FrequencyBand(10.0, 5)),
                  sponge_cells=sponge)
    params.update(overrides)
    return FWIConfig(**params)
