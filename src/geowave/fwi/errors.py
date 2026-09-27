from __future__ import annotations
import numpy as np

class FWIAbortError(RuntimeError):
    """Inversion aborted; the message carries the human-readable reason."""

def require_finite(values: np.ndarray, what: str, context: str) -> None:
    a = np.asarray(values)
    if not np.isfinite(a).all():
        bad = int(a.size - np.count_nonzero(np.isfinite(a)))
        raise FWIAbortError(f'{what}: {bad} non-finite value(s) detected ({context})')
