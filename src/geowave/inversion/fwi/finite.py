from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NonFiniteDetail:
    stage: str
    kind: str
    count: int
    first_index: tuple[int, ...] | None

    def describe(self) -> str:
        where = '' if self.first_index is None else f', first at index {self.first_index}'
        return f'{self.kind} values in {self.stage} ({self.count} non-finite samples{where})'


class FWINonFiniteError(RuntimeError):
    """Raised when data, a wavefield, a residual or a gradient leaves the finite reals."""

    def __init__(self, detail: NonFiniteDetail, result: object | None = None):
        super().__init__(detail.describe())
        self.detail = detail
        self.result = result


def finite_detail(name: str, a: np.ndarray) -> NonFiniteDetail | None:
    a = np.asarray(a)
    mask = ~np.isfinite(a)
    if not mask.any():
        return None
    count = int(mask.sum())
    idx = np.unravel_index(int(np.flatnonzero(mask.ravel())[0]), a.shape)
    kind = 'NaN' if np.isnan(a).any() else 'infinite'
    return NonFiniteDetail(name, kind, count, tuple(int(i) for i in idx))


def assert_finite(name: str, a: np.ndarray) -> None:
    detail = finite_detail(name, np.asarray(a))
    if detail is not None:
        raise FWINonFiniteError(detail)
