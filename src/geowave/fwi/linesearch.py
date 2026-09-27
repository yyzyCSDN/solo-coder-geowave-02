from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LineSearchOutcome:
    accepted: bool
    step: float
    objective: float
    trials: int
    model: np.ndarray
    message: str

def armijo_backtracking(evaluate, model: np.ndarray, gradient: np.ndarray, direction: np.ndarray,
                        objective0: float, initial_step: float, vmin: float, vmax: float,
                        c: float=1e-4, shrink: float=0.5, max_backtracks: int=8,
                        on_trial=None) -> LineSearchOutcome:
    """Projected Armijo backtracking on the box constraint [vmin, vmax].

    `evaluate(m)` must return the misfit and may raise FWIAbortError.
    The input `model` is never modified: on failure the previous model is
    returned unchanged (rollback), on success the accepted trial model.
    """
    gtd = float(np.sum(gradient * direction))
    if gtd >= 0.0:
        return LineSearchOutcome(False, 0.0, objective0, 0, model,
                                 f'not a descent direction (g.d={gtd:.3e}); rolled back')
    step = initial_step
    last_objective = objective0
    for trial in range(1, max_backtracks + 1):
        candidate = np.clip(model + step * direction, vmin, vmax)
        objective = float(evaluate(candidate))
        last_objective = objective
        sufficient = objective0 + c * step * gtd
        if on_trial is not None:
            on_trial(trial, step, objective, objective <= sufficient)
        if objective <= sufficient:
            return LineSearchOutcome(True, step, objective, trial, candidate, 'accepted')
        step *= shrink
    return LineSearchOutcome(False, 0.0, objective0, max_backtracks, model,
                             f'no sufficient decrease after {max_backtracks} trials '
                             f'(last objective {last_objective:.6e}); rolled back')
