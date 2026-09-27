from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt

def lowpass_zero_phase(data: np.ndarray, dt_s: float, cutoff_hz: float, order: int=4, axis: int=0) -> np.ndarray:
    """Zero-phase Butterworth lowpass along `axis` (time). Used to build the
    frequency bands of the multi-scale cascade; applied identically to the
    observed data and the source wavelet so both stay in the same band."""
    a = np.asarray(data, float)
    ny = 0.5 / dt_s
    if not 0 < cutoff_hz < ny:
        raise ValueError(f'cutoff {cutoff_hz:g} Hz outside (0, {ny:g}) Hz')
    sos = butter(order, cutoff_hz / ny, btype='low', output='sos')
    return sosfiltfilt(sos, a, axis=axis)
