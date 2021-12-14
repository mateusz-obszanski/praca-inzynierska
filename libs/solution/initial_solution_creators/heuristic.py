"""
Heuristic solution creators.
"""

from typing import TypeVar, Union

import numpy as np
from scipy.optimize import dual_annealing

from libs.utils.graph import graph_cycle_greedy_nn
from libs.environment.cost_calculators import CostT, DynCosts, cost_calc_irp


Rng = TypeVar("Rng", bound=np.random.Generator)


def create_tsp_sol_nn(
    dist_mx: np.ndarray, initial_vx: int, forbidden_val: float
) -> list[int]:
    # FIXME vvv does not return a cycle - missing initial_vx at the end
    # for grid it doesn't matter
    return graph_cycle_greedy_nn(dist_mx, initial_vx, forbidden_val)


def create_irp_sol_nn_sa(
    ext_dist_mx: np.ndarray,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
    dyn_costs: DynCosts,
    vehicle_volume: float,
    vehicle_speed: float,
    demands: tuple[float],
    w1: float,
    w2: float,
    rng: Rng,
) -> tuple[list[int], list[float], CostT, Rng]:
    route = graph_cycle_greedy_nn(ext_dist_mx, initial_vx, forbidden_val)
    route_len = len(route)
    low = np.full(shape=route_len, fill_value=0.0, dtype=np.float64)
    high = np.full(shape=route_len, fill_value=vehicle_volume, dtype=np.float64)
    bounds = np.stack((low, high), axis=1).tolist()
    cost_func = lambda _quantities: cost_calc_irp(
        route,
        dyn_costs,
        ext_dist_mx,
        vehicle_speed,
        initial_vx,
        forbidden_val,
        ini_and_dummy_vxs,
        demands,
        w1,
        w2,
        _quantities.tolist(),
    )
    result = dual_annealing(cost_func, bounds, seed=rng)
    sol_qs = result.x
    cost = result.fun
    return route, sol_qs, cost, rng
