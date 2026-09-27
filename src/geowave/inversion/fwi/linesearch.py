from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.inversion.fwi.finite import assert_finite, finite_detail
from geowave.inversion.fwi.grid import clamp_velocity


@dataclass(frozen=True)
class LineSearchResult:
    accepted: bool
    step: float
    objective: float
    model: np.ndarray
    evaluations: tuple[dict, ...]
    reason: str = ''


def backtracking_line_search(
    model: np.ndarray,
    direction: np.ndarray,
    objective0: float,
    grad0: np.ndarray,
    evaluate,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    initial_step: float = 100.0,
    shrink: float = 0.5,
    max_trials: int = 12,
    c1: float = 1e-4,
) -> LineSearchResult:
    """Backtracking Armijo line search with hard velocity bounds and full rollback.

    ``evaluate(model) -> float`` returns the objective for a trial model. Trials
    that produce non-finite values never touch the caller's model; the step is
    simply rejected and shrunk. If every trial fails, the original model and
    objective are returned with ``accepted=False``.
    """
    m0 = np.array(model, dtype=float)
    d = np.asarray(direction, dtype=float)
    assert_finite('search direction', d)
    directional = float(np.sum(grad0 * d))
    if directional >= 0.0:
        return LineSearchResult(False, 0.0, objective0, m0, (),
                                reason=f'descent check failed: g.d = {directional:.3e} >= 0')

    evaluations: list[dict] = []
    step = float(initial_step)
    for trial in range(max_trials):
        trial_model = clamp_velocity(m0 + step * d, vmin, vmax)
        detail = finite_detail(f'trial model (step={step:g})', trial_model)
        obj: float = np.inf
        if detail is None:
            try:
                obj = float(evaluate(trial_model))
            except Exception as exc:  # trial failure must not abort the run
                evaluations.append({'trial': trial, 'step': step, 'objective': None,
                                    'rejected': f'evaluate raised {type(exc).__name__}: {exc}'})
                step *= shrink
                continue
        if detail is not None:
            evaluations.append({'trial': trial, 'step': step, 'objective': None,
                                'rejected': detail.describe()})
        elif not np.isfinite(obj):
            kind = 'NaN' if np.isnan(obj) else 'infinite'
            evaluations.append({'trial': trial, 'step': step, 'objective': None,
                                'rejected': f'non-finite objective ({kind})'})
        else:
            armijo = objective0 + c1 * step * directional
            record = {'trial': trial, 'step': step, 'objective': obj,
                      'armijo_target': armijo}
            if obj <= armijo:
                record['accepted'] = True
                evaluations.append(record)
                return LineSearchResult(True, step, obj, trial_model,
                                        tuple(evaluations))
            record['rejected'] = 'Armijo condition not met'
            evaluations.append(record)
        step *= shrink
    return LineSearchResult(False, 0.0, objective0, m0, tuple(evaluations),
                            reason=f'no acceptable step after {max_trials} trials')
