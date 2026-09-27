"""Time-domain 2-D acoustic full-waveform inversion (FWI) toolkit.

Multiscale low-to-high frequency continuation, adjoint-state gradients,
hard velocity bounds and a rollback backtracking line search. Runs leave an
objective history (history.jsonl) and model snapshots (snapshots/iter_*.npz)
via RunArchive; non-finite data, wavefields or gradients abort with
FWINonFiniteError naming the failing stage.
"""
from geowave.inversion.fwi.archive import RunArchive
from geowave.inversion.fwi.filters import FrequencyBand, continuation_bands
from geowave.inversion.fwi.finite import FWINonFiniteError
from geowave.inversion.fwi.grid import GridSpec, stable_dt
from geowave.inversion.fwi.run import FWIConfig, run_fwi
from geowave.inversion.fwi.survey import ShotGeometry, Survey, surface_survey
from geowave.inversion.fwi.synthetic import generate_synthetic
from geowave.inversion.fwi.wave import AbsorbingBoundary

__all__ = [
    'RunArchive', 'FrequencyBand', 'continuation_bands', 'FWINonFiniteError',
    'GridSpec', 'stable_dt', 'FWIConfig', 'run_fwi', 'ShotGeometry', 'Survey',
    'surface_survey', 'generate_synthetic', 'AbsorbingBoundary',
]
