from typing import Generator, Protocol, Sequence
from math import isfinite

import numpy as np
import more_itertools as mit

from libs.solution import SolutionTSP
from libs.utils.iteration import split_list_mult


CostT = float
CostGen = Generator[CostT, None, None]

CostMx = np.ndarray
ExpirationTimeDelta = float
DistMx = np.ndarray
DynCosts = Sequence[tuple[CostMx, ExpirationTimeDelta]]


class CostCalculator(Protocol):
    def __call__(
        self,
        vx_seq: Sequence,
        dyn_costs: DynCosts,
        distance_mx: DistMx,
        salesman_v: float,
        *args,
        forbidden_val: float = -1,
        **kwargs,
    ) -> CostT:
        ...


def cost_calc_tsp(
    vx_seq: Sequence[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    forbidden_val: float = -1,
) -> float:
    present_vxs = set(vx_seq)
    all_vxs = range(len(distance_mx))
    if any(vx not in present_vxs for vx in all_vxs):
        return float("inf")
    total_t = 0
    window_total_t = 0
    dyn_costs_iter = iter(dyn_costs)
    time_mx, expir_t_delta = next(dyn_costs_iter)
    for vx1, vx2 in mit.windowed(vx_seq, n=2):
        dist = distance_mx[vx1, vx2]
        expected_t_delta = time_mx[vx1, vx2]
        if not isfinite(dist) or dist == forbidden_val:
            return float("inf")
        end_t = window_total_t + expected_t_delta
        if end_t > expir_t_delta:
            # calc time at time windows border
            to_window_end = expir_t_delta - window_total_t
            traversed_dist = salesman_v / to_window_end
            dist_rest = dist - traversed_dist
            next_window_t_delta = salesman_v / dist_rest
            total_t += to_window_end + next_window_t_delta
            window_total_t = next_window_t_delta
            try:
                time_mx, expir_t_delta = next(dyn_costs_iter)
            except StopIteration:
                return total_t
        else:
            total_t += end_t
            window_total_t += end_t
    return total_t


def cost_calc_vrp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    ini_and_dummy_vxs: set[int],
    forbidden_val: float = -1,
) -> float:
    drone_seqs, _ = split_list_mult(vx_seq, ini_and_dummy_vxs)
    # ^^ splits at 0, so the first and the last subroutes is empty
    drone_seqs = ((0, *sr, 0) for sr in drone_seqs[1:-1])
    return max(
        cost_calc_tsp(ds, dyn_costs, distance_mx, salesman_v, forbidden_val)
        for ds in drone_seqs
    )


def cost_calc_sdvrp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    ini_and_dummy_vxs: set[int],
    demands: tuple[int],
    w1: float,
    w2: float,
    forbidden_val: float = -1,
) -> float:
    rewards = (
        0 if d in ini_and_dummy_vxs else max(d, sum(1 for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    return w1 * cost_calc_vrp(
        vx_seq, dyn_costs, distance_mx, salesman_v, ini_and_dummy_vxs, forbidden_val
    ) - w2 * sum(rewards)


def cost_calc_irp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    ini_and_dummy_vxs: set[int],
    demands: tuple[float],
    w1: float,
    w2: float,
    quantities: list[float],
    forbidden_val: float = -1,
) -> float:
    assert len(vx_seq) == len(demands)
    rewards = (
        0
        if d in ini_and_dummy_vxs
        else max(d, sum(quantities[i] for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    return w1 * cost_calc_vrp(
        vx_seq, dyn_costs, distance_mx, salesman_v, ini_and_dummy_vxs, forbidden_val
    ) - w2 * sum(rewards)


class CostGenCreator(Protocol):
    """
    Assumes that solution does not start from initial vx.
    """

    def __call__(self, solution, cost_mx: np.ndarray, initial_ix: int = 0) -> CostGen:
        """
        Assumes that solution does not start from initial vx.
        """
        ...


def calculate_total_cost(cost_generator: CostGen) -> CostT:
    return sum(cost_generator)


def cost_gen_tsp(
    solution: SolutionTSP, cost_mx: np.ndarray, initial_ix: int = 0
) -> CostGen:
    """
    Assumes that solution doesn't start from initial vx.
    """
    yield cost_mx[initial_ix, solution[0]]

    yield from (
        cost_mx[current_vx, next_vx]
        for current_vx, next_vx in mit.windowed(solution, n=2)
    )
