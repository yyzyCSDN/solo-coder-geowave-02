from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class FrequencyBand:
    """Zero-phase pass band used for one FWI scale (frequencies in Hz).

    Low-frequency stages use a low-cut of 0 (low-pass); higher stages become
    band-pass, giving the low-to-high continuation.
    """

    f_low_hz: float
    f_high_hz: float
    label: str = ''

    def __post_init__(self):
        if self.f_low_hz < 0 or self.f_high_hz <= self.f_low_hz:
            raise ValueError('require 0 <= f_low < f_high')

    def __str__(self) -> str:
        return self.label or f'{self.f_low_hz:g}-{self.f_high_hz:g}Hz'


def continuation_bands(f_start_hz: float, f_end_hz: float,
                       factor: float = 2.0) -> tuple[FrequencyBand, ...]:
    """Geometric sequence of widening bands [0, f_start], [0, f_start*factor], ..."""
    if f_start_hz <= 0 or f_end_hz < f_start_hz or factor <= 1:
        raise ValueError('require 0 < f_start <= f_end and factor > 1')
    bands = []
    f = f_start_hz
    while f < f_end_hz:
        bands.append(FrequencyBand(0.0, f, f'0-{f:g}Hz'))
        f *= factor
    bands.append(FrequencyBand(0.0, float(f_end_hz), f'0-{f_end_hz:g}Hz'))
    return tuple(bands)


def bandpass(data: np.ndarray, dt_s: float, band: FrequencyBand,
             order: int = 4, axis: int = 0) -> np.ndarray:
    """Zero-phase Butterworth filtering of gathers (low-pass if f_low == 0)."""
    nyq = 0.5 / dt_s
    high = min(band.f_high_hz, nyq * (1.0 - 1e-6))
    if high <= 0:
        return np.zeros_like(data)
    if band.f_low_hz > 0:
        low = min(band.f_low_hz, high * (1.0 - 1e-6))
        sos = butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')
    else:
        sos = butter(order, high / nyq, btype='lowpass', output='sos')
    pad = min(data.shape[axis] - 1, 3 * (2 * order))
    if pad < 3:
        return data.copy()
    return sosfiltfilt(sos, data, axis=axis, padlen=pad)
