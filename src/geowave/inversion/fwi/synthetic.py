from __future__ import annotations
import numpy as np
from geowave.inversion.fwi.finite import assert_finite
from geowave.inversion.fwi.grid import GridSpec, stable_dt
from geowave.inversion.fwi.survey import Survey, SyntheticData
from geowave.inversion.fwi.wave import AbsorbingBoundary, forward_model
from geowave.model.wavelet import Wavelet


def generate_synthetic(velocity: np.ndarray, spec: GridSpec, survey: Survey,
                       wavelet: Wavelet, record_time_s: float,
                       abc: AbsorbingBoundary | None = None,
                       *, dt_s: float | None = None) -> SyntheticData:
    """Generate common-shot gathers over ``velocity`` (the 'true' model).

    Raises FWINonFiniteError if the source, any wavefield or any gather leaves
    the finite reals.
    """
    v = np.asarray(velocity, dtype=float)
    if v.shape != spec.shape:
        raise ValueError('velocity shape does not match grid')
    if dt_s is None:
        dt_s = min(stable_dt(spec, float(np.max(v))), wavelet.dt_s)
    nt = int(np.ceil(record_time_s / dt_s))
    # Resample the source signature onto the simulation time axis.
    src_t = np.arange(wavelet.samples.size) * wavelet.dt_s
    t = np.arange(nt) * dt_s
    source = np.interp(t, src_t - src_t[0], wavelet.samples, left=0.0, right=0.0)
    assert_finite('source wavelet', source)

    gathers = np.zeros((len(survey), nt, survey.shots[0].n_receivers))
    for k, shot in enumerate(survey):
        result = forward_model(v, spec, dt_s, nt, source, shot.source_xz,
                               shot.receiver_xz, abc, store_wavefield=False)
        gathers[k] = result.data
    assert_finite('synthetic dataset', gathers)
    return SyntheticData(velocity=v, shots=tuple(survey.shots), gathers=gathers,
                         dt_s=float(dt_s), wavelet_name=wavelet.name)
