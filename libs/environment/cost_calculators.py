from typing import Generator, Protocol
from collections.abc import Iterator, Sequence
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
        initial_vx: int,
        forbidden_val: float,
        *args,
        **kwargs,
    ) -> CostT:
        ...


# def cost_calc_tsp(
#     vx_seq: Sequence[int],
#     dyn_costs: DynCosts,
#     distance_mx: DistMx,
#     salesman_v: float,
#     forbidden_val: float,
# ) -> float:
#     present_vxs = set(vx_seq)
#     all_vxs = range(len(distance_mx))
#     if any(vx not in present_vxs for vx in all_vxs):
#         return float("inf")
#     total_t = 0
#     window_total_t = 0
#     dyn_costs_iter = iter(dyn_costs)
#     time_mx, expir_t_delta = next(dyn_costs_iter)
#     for vx1, vx2 in mit.windowed(vx_seq, n=2):
#         dist = distance_mx[vx1, vx2]
#         expected_t_delta = time_mx[vx1, vx2]
#         if not isfinite(dist) or dist == forbidden_val:
#             return float("inf")
#         end_t = window_total_t + expected_t_delta
#         traversed_dist = 0
#         """
#         jeżeli przewidywany czas dolotu przekracza exp_t:
#             przesuń się o dystans, który jest w stanie pokonać
#             następne okno i exp_t
#             zapamiętaj pokonany dystans
#             to samo dla następnej macierzy
#         """
#         while end_t > expir_t_delta:
#             # calc time at time windows border
#             to_window_end = expir_t_delta - window_total_t
#             traversed_dist += salesman_v / to_window_end
#             dist_rest = dist - traversed_dist
#             next_window_t_delta = salesman_v / dist_rest
#             total_t += to_window_end + next_window_t_delta
#             # time and distance already traversed for the current time matrix
#             window_total_t = next_window_t_delta
#             try:
#                 time_mx, expir_t_delta = next(dyn_costs_iter)
#             except StopIteration:
#                 return total_t
#         else:
#             total_t += end_t
#             window_total_t += end_t
#     return total_t


def cost_calc_tsp(
    vx_seq: Sequence[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    initial_vx: int,
    forbidden_val: float,
) -> float:
    present_vxs = set(vx_seq)
    all_vxs = range(len(distance_mx))
    if any(vx not in present_vxs for vx in all_vxs):
        return float("inf")
    return cost_calc_core(
        vx_seq, dyn_costs, distance_mx, salesman_v, initial_vx, forbidden_val
    )


def cost_calc_core(
    vx_seq: Sequence[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    initial_vx: int,
    forbidden_val: float,
) -> float:
    if initial_vx != vx_seq[0] != vx_seq[-1] != initial_vx:
        return float("inf")
    total_t = 0
    vx_pair_iter: Iterator[tuple[int, int]] = mit.windowed(vx_seq, n=2)  # type: ignore
    try:
        v1, v2 = next(vx_pair_iter)
    except StopIteration:
        return 0
    for travel_time, mx_exp_t in dyn_costs:
        window_t = 0
        # distance already traversed between v1 and v2
        traversed_d = 0
        dist_v1_v2: float = distance_mx[v1, v2]
        if not isfinite(dist_v1_v2) or dist_v1_v2 == forbidden_val:
            return float("inf")
        travel_t_v1_v2: float = travel_time[v1, v2]
        # salesman effective speed taking into account the wind
        salesman_eff_v = dist_v1_v2 / travel_t_v1_v2
        while True:
            dist_to_go = dist_v1_v2 - traversed_d
            # destination reaching expected time for current travel_time matrix
            expected_end_t = total_t + dist_to_go / salesman_eff_v
            if expected_end_t <= mx_exp_t:
                traversed_d += dist_to_go
                window_t += expected_end_t
                total_t += expected_end_t
                try:
                    v1, v2 = next(vx_pair_iter)
                except StopIteration:
                    return total_t
                dist_v1_v2 = distance_mx[v1, v2]
                if not isfinite(dist_v1_v2) or dist_v1_v2 == forbidden_val:
                    return float("inf")
                travel_t_v1_v2 = travel_time[v1, v2]
                salesman_eff_v = dist_v1_v2 / travel_t_v1_v2
                traversed_d = 0
                break
            else:
                time_to_end = mx_exp_t - window_t
                total_t += time_to_end
                traversed_d += salesman_v / time_to_end
                window_t = 0
    return total_t


def cost_calc_vrp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
) -> float:
    drone_seqs, _ = split_list_mult(vx_seq, ini_and_dummy_vxs)
    # ^^ splits at 0, so the first and the last subroutes is empty
    drone_seqs = ((0, *sr, 0) for sr in drone_seqs[1:-1])
    return max(
        cost_calc_tsp(ds, dyn_costs, distance_mx, salesman_v, initial_vx, forbidden_val)
        for ds in drone_seqs
    )


def cost_calc_sdvrp_core(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
) -> float:
    drone_seqs, _ = split_list_mult(vx_seq, ini_and_dummy_vxs)
    # ^^ splits at 0, so the first and the last subroutes is empty
    drone_seqs = ((0, *sr, 0) for sr in drone_seqs[1:-1])
    return max(
        cost_calc_core(
            ds, dyn_costs, distance_mx, salesman_v, initial_vx, forbidden_val
        )
        for ds in drone_seqs
    )


def cost_calc_sdvrp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    forbidden_val: float,
    initial_vx: int,
    ini_and_dummy_vxs: set[int],
    demands: tuple[int],
    w1: float,
    w2: float,
) -> float:
    rewards = (
        0 if d in ini_and_dummy_vxs else max(d, sum(1 for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    return (
        w1
        * cost_calc_sdvrp_core(
            vx_seq,
            dyn_costs,
            distance_mx,
            salesman_v,
            initial_vx,
            ini_and_dummy_vxs=ini_and_dummy_vxs,
            forbidden_val=forbidden_val,
        )
        - w2 * sum(rewards)
    )


def cost_calc_irp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    salesman_v: float,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
    demands: tuple[float],
    w1: float,
    w2: float,
    quantities: list[float],
) -> float:
    assert len(vx_seq) == len(demands)
    rewards = (
        0
        if d in ini_and_dummy_vxs
        else max(d, sum(quantities[i] for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    return (
        w1
        * cost_calc_sdvrp_core(
            vx_seq,
            dyn_costs,
            distance_mx,
            salesman_v,
            initial_vx,
            ini_and_dummy_vxs=ini_and_dummy_vxs,
            forbidden_val=forbidden_val,
        )
        - w2 * sum(rewards)
    )


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
