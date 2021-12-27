from typing import Any, Callable, Optional, TypeVar
from math import exp, pi

import numpy as np
from scipy.stats import rv_continuous


Rng = TypeVar("Rng", bound=np.random.Generator)


def simulated_annealing(
    f: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    rng: Rng,
    step_max: float = 1,
    iter_max: int = 1000,
    initial_temp: float = 5230,
    initial_sol: Optional[np.ndarray] = None,
    early_stop_iters: Optional[int] = None,
) -> tuple[np.ndarray, float, dict[str, Any], Rng]:
    """
    `f` - objective function to minimize,
    `bounds` - ndarray with shape (objective vector length, 2) - pairs (min, max)

    Implemented features:
    - early stop after `early_stop_iters` without progress
    """

    if early_stop_iters is None:
        early_stop_iters = iter_max

    if initial_sol is None:
        sol_best = bounds[:, 0] + rng.random(size=len(bounds)) * (
            bounds[:, 1] - bounds[:, 0]
        )
    else:
        sol_best = initial_sol
    obj_curr = obj_best = f(sol_best)
    sol_curr = sol_best
    i = 0
    since_better = 0
    for i in range(1, iter_max + 1):
        # candidate
        cand = sol_curr + rng.uniform(-1, 1, size=len(bounds)) * step_max
        obj_cand = f(cand)
        if obj_cand < obj_best:
            sol_best, obj_best = cand, obj_cand
        obj_diff = obj_cand - obj_curr
        temp = initial_temp / i
        # metropolis acceptance criterion
        metr_crit_val = exp(-obj_diff / temp)
        if obj_diff < 0 or rng.random() < metr_crit_val:
            since_better = 0
            sol_curr = cand
            obj_curr = obj_cand
        else:
            since_better += 1
        if since_better >= early_stop_iters:
            break
    info = {
        "iterations": i
    }
    return sol_best, obj_best, info, rng


def generalized_simulated_annealing(
    f: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    rng: Rng,
    iter_max: int = 1000,
    initial_temp: float = 5230,
    initial_sol: Optional[np.ndarray] = None,
    early_stop_iters: Optional[int] = None,
    alpha: float = 0.9,
    beta: float = -20,
    a: float = 1e-3,
) -> tuple[np.ndarray, float, dict[str, Any], Rng]:
    """
    based on: https://www.intechopen.com/chapters/52820
    and: http://rbanchs.com/documents/THFEL_PR15.pdf

    `f` - objective function to minimize,
    `bounds` - ndarray with shape (objective vector length, 2) - pairs (min, max)
    `alpha` - shape parameter for cooling schedule
    `beta` - shape parameter for exponential distortion
    `a` - shape parameter for Cauchy probabiity distribution function

    Implemented features:
    - early stop after `early_stop_iters` without progress
    - exponential cooling schedule
    - exponential distortion
    """

    assert beta > 0

    random_unit = rng.random

    # number of dimensions
    D = len(bounds)
    D_inv = 1 / D
    alpha_neg = -alpha
    beta_neg = -beta

    if early_stop_iters is None:
        early_stop_iters = iter_max

    if initial_sol is None:
        sol_best = bounds[:, 0] + random_unit(size=D) * (
            bounds[:, 1] - bounds[:, 0]
        )
    else:
        sol_best = initial_sol
    obj_curr = obj_best = f(sol_best)
    sol_curr = sol_best
    obj_curr_distorted = obj_best_distorted = -exp(beta_neg * obj_best)
    i = 0
    since_better = 0
    cauchy_gen = CauchyDistr(seed=rng.integers((1 << 32) - 1))
    temp = initial_temp
    for i in range(1, iter_max + 1):
        # candidate
        cand = sol_curr + cauchy_gen.rvs(a=a, T=temp, size=D)
        obj_cand = f(cand)
        if obj_cand < obj_best:
            sol_best = cand
            obj_best = obj_cand
        obj_diff = obj_cand - obj_curr
        obj_diff_distorted = -exp(beta_neg * obj_diff)

        temp = initial_temp * exp(alpha_neg * i**D_inv)
        # exponential distortion
        # metropolis acceptance criterion
        metr_crit_val = exp(-obj_diff_distorted / temp)
        if obj_diff < 0 or random_unit() < metr_crit_val:
            since_better = 0
            sol_curr = cand
            obj_curr = obj_cand
        else:
            since_better += 1
        if since_better >= early_stop_iters:
            break
    info = {
        "iterations": i
    }
    return sol_best, obj_best, info, rng


class CauchyDistr(rv_continuous):
    "Cauchy distribution for simulated annealing"
    def _pdf(self, x, a: float, T: float):
        aT = a*T
        return aT / (pi*aT*aT + x*x)
