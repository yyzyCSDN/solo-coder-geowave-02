from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import math

@dataclass(frozen=True)
class FrequencyBand:
    """One scale of the multi-scale cascade: data is lowpassed at `lowpass_hz`."""
    lowpass_hz: float
    max_iterations: int

    def __post_init__(self):
        if self.lowpass_hz <= 0:
            raise ValueError('lowpass_hz must be positive')
        if self.max_iterations < 1:
            raise ValueError('max_iterations must be >= 1')

@dataclass(frozen=True)
class FWIConfig:
    nx: int
    nz: int
    dx_m: float
    tmax_s: float
    shot_x_m: tuple[float, ...]
    receiver_x_m: tuple[float, ...]
    vmin_mps: float
    vmax_mps: float
    bands: tuple[FrequencyBand, ...]
    wavelet_peak_hz: float = 6.0
    source_z_m: float = 20.0
    receiver_z_m: float = 20.0
    sponge_cells: int = 15
    free_surface: bool = True
    cfl_safety: float = 0.5
    dt_s: float | None = None
    initial_step_fraction: float = 0.05
    armijo_c: float = 1e-4
    step_shrink: float = 0.5
    max_backtracks: int = 8
    gradient_smoothing_cells: int = 2
    illumination_preconditioner: bool = True
    illumination_epsilon: float = 0.1
    save_gradient_snapshots: bool = True

    def __post_init__(self):
        if self.nx < 8 or self.nz < 8:
            raise ValueError('grid too small')
        if not 0 < self.vmin_mps < self.vmax_mps:
            raise ValueError('require 0 < vmin < vmax')
        if not self.bands:
            raise ValueError('at least one frequency band required')
        cuts = [b.lowpass_hz for b in self.bands]
        if cuts != sorted(cuts):
            raise ValueError('bands must be ordered low to high frequency')
        if not (0.0 < self.initial_step_fraction <= 1.0):
            raise ValueError('initial_step_fraction in (0, 1]')
        if not (0.0 < self.step_shrink < 1.0):
            raise ValueError('step_shrink in (0, 1)')

    @property
    def dt(self) -> float:
        """Time step: user-supplied or derived from the CFL limit at vmax."""
        if self.dt_s is not None:
            return self.dt_s
        return self.cfl_safety / (self.vmax_mps * math.sqrt(2.0) / self.dx_m)

    @property
    def nt(self) -> int:
        return int(round(self.tmax_s / self.dt)) + 1

    def cfl_ok(self) -> bool:
        limit = 1.0 / (self.vmax_mps * math.sqrt(2.0) / self.dx_m)
        return self.dt <= limit

    def points_per_wavelength(self) -> float:
        """Worst-case (slowest velocity, highest frequency) sampling ratio."""
        fmax = self.bands[-1].lowpass_hz
        return self.vmin_mps / (fmax * self.dx_m)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @staticmethod
    def parse_bands(spec: str) -> tuple[FrequencyBand, ...]:
        """Parse 'cutoff:iters,cutoff:iters' e.g. '2.5:5,5:5,10:8'."""
        bands = []
        for part in spec.split(','):
            cut, _, iters = part.strip().partition(':')
            bands.append(FrequencyBand(float(cut), int(iters) if iters else 5))
        return tuple(bands)
