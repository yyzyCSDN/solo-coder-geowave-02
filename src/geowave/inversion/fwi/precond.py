from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_gradient(gradient: np.ndarray, grid_spacing_m: tuple[float, float],
                    radius_m: float) -> np.ndarray:
    """Gaussian preconditioning / anti-ringing (radius 0 disables)."""
    if radius_m <= 0:
        return np.asarray(gradient, dtype=float)
    sigma = (radius_m / max(grid_spacing_m[0], 1e-12),
             radius_m / max(grid_spacing_m[1], 1e-12))
    return gaussian_filter(np.asarray(gradient, dtype=float), sigma=sigma)


def descent_direction(gradient: np.ndarray, spec, smooth_radius_m: float = 0.0) -> np.ndarray:
    """Steepest-descent direction -g/max|g|, optionally pre-smoothed.

    Max-norm scaling makes the line-search step an interpretable per-cell
    velocity update in m/s (the strongest cell moves by exactly ``step``).
    """
    g = smooth_gradient(np.asarray(gradient, dtype=float),
                        (spec.dz, spec.dx), smooth_radius_m)
    scale = float(np.max(np.abs(g)))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(g)
    return -g / scale
