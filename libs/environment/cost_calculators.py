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
NonNormObjVec = list[CostT]
"""
Vector of cost/reward function values to be normalized by `normalize_obj_min`
or `normalize_obj_max` respectively.
"""


def normalize_obj_min(current: float, highest: float, lowest: float) -> float:
    """
    Returns normalized objective function to be minimized.
    """

    return (highest - current) / (highest - lowest)


def normalize_obj_max(current: float, highest: float, lowest: float) -> float:
    """
    Returns normalized objective function to be maximized.
    """

    return 1 - (highest - current) / (highest - lowest)


class CostCalculator(Protocol):
    def __call__(
        self,
        vx_seq: Sequence,
        dyn_costs: DynCosts,
        distance_mx: DistMx,
        initial_vx: int,
        forbidden_val: float,
        *args,
        **kwargs,
    ) -> NonNormObjVec:
        """
        Highest and lowest costs estimations or current found values - for normalization.
        """
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
    initial_vx: int,
    forbidden_val: float,
) -> NonNormObjVec:
    present_vxs = set(vx_seq)
    all_vxs = range(len(distance_mx))
    if any(vx not in present_vxs for vx in all_vxs):
        return [float("inf")]
    return [cost_calc_core(vx_seq, dyn_costs, distance_mx, initial_vx, forbidden_val)]


def cost_calc_core(
    vx_seq: Sequence[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
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
    EPS = 1e-4
    for travel_time, mx_exp_t in dyn_costs:
        # distance already traversed between v1 and v2
        dist_traversed_v1_v2 = 0
        dist_v1_v2: float = distance_mx[v1, v2]
        if not isfinite(dist_v1_v2) or dist_v1_v2 == forbidden_val:
            return float("inf")
        travel_t_v1_v2: float = travel_time[v1, v2]
        # for security - zero division
        while travel_t_v1_v2 < EPS:
            try:
                v1, v2 = next(vx_pair_iter)
            except StopIteration:
                if v2 == initial_vx:
                    return total_t
                return float("inf")
            dist_v1_v2 = distance_mx[v1, v2]
            if not isfinite(dist_v1_v2) or dist_v1_v2 == forbidden_val:
                return float("inf")
            travel_t_v1_v2 = travel_time[v1, v2]
        # salesman effective speed taking into account the wind
        salesman_eff_v = dist_v1_v2 / travel_t_v1_v2
        while True:
            if -EPS < salesman_eff_v < EPS or salesman_eff_v <= 0:
                # cannot move during this time window
                break
            dist_to_go = dist_v1_v2 - dist_traversed_v1_v2

            # destination reaching expected time for current travel_time matrix
            expected_end_t = total_t + dist_to_go / salesman_eff_v
            if expected_end_t >= mx_exp_t:
                # didn't reach destination in this time window
                time_to_end = mx_exp_t - total_t
                dist_traversed_v1_v2 += salesman_eff_v * time_to_end
                total_t += time_to_end
                break
            else:
                # reached destination, there's still time
                total_t += dist_to_go / salesman_eff_v

                # next vertex pair setup
                # for security - zero division
                travel_t_v1_v2 = -1
                while travel_t_v1_v2 < EPS:
                    try:
                        v1, v2 = next(vx_pair_iter)
                    except StopIteration:
                        if v2 == initial_vx:
                            return total_t
                        return float("inf")
                    dist_v1_v2 = distance_mx[v1, v2]
                    if not isfinite(dist_v1_v2) or dist_v1_v2 == forbidden_val:
                        return float("inf")
                    travel_t_v1_v2 = travel_time[v1, v2]
                salesman_eff_v = dist_v1_v2 / travel_t_v1_v2
                dist_traversed_v1_v2 = 0

    try:
        next(vx_pair_iter)
        return float("inf")
    except StopIteration:
        # if v2 == initial_vx:  # already checked at the beginning
        #     return total_t
        # return float("inf")
        return total_t


def cost_calc_vrp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
) -> NonNormObjVec:
    """
    Distance matrix must be extended, the same for dyn_costs.
    """
    present_vxs = set(vx_seq) - ini_and_dummy_vxs
    all_vxs = set(range(len(distance_mx)))
    if not all_vxs - (present_vxs | {initial_vx}):
        # some vertices not present in the solution
        return [float("inf")]
    drone_seqs, _ = split_list_mult(vx_seq, ini_and_dummy_vxs)
    # ^^ splits at 0, so the first and the last subroutes is empty
    drone_seqs = ((0, *sr, 0) for sr in drone_seqs[1:-1])
    return [
        max(
            cost_calc_core(ds, dyn_costs, distance_mx, initial_vx, forbidden_val)
            for ds in drone_seqs
        )
    ]


def cost_calc_vrpp_core(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
) -> float:
    """
    Distance matrix must be extended, the same for dyn_costs.
    """
    drone_seqs, _ = split_list_mult(vx_seq, ini_and_dummy_vxs)
    # ^^ splits at 0, so the first and the last subroutes is empty
    drone_seqs = ((0, *sr, 0) for sr in drone_seqs[1:-1])
    return max(
        cost_calc_core(ds, dyn_costs, distance_mx, initial_vx, forbidden_val)
        for ds in drone_seqs
    )


def cost_calc_vrpp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
    demands: tuple[int],
    fillval: int,
) -> NonNormObjVec:
    """
    Distance matrix must be extended, the same for dyn_costs. Ignores fillvals.
    """

    vx_seq = [vx for vx in vx_seq if vx != fillval]
    rewards = (
        0 if d in ini_and_dummy_vxs else max(d, sum(1 for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    cost = cost_calc_vrpp_core(
        vx_seq,
        dyn_costs,
        distance_mx,
        initial_vx,
        forbidden_val=forbidden_val,
        ini_and_dummy_vxs=ini_and_dummy_vxs,
    )
    reward = sum(rewards)
    return [cost, reward]


def cost_calc_irp(
    vx_seq: list[int],
    dyn_costs: DynCosts,
    distance_mx: DistMx,
    initial_vx: int,
    forbidden_val: float,
    ini_and_dummy_vxs: set[int],
    demands: tuple[float],
    fillval: int,
    quantities: list[float],
) -> NonNormObjVec:
    """
    Distance matrix must be extended, the same for dyn_costs.
    """
    assert len(vx_seq) == len(demands)
    filler_ixs = set(ix for ix, vx in enumerate(vx_seq) if vx == fillval)
    vx_seq = [vx for ix, vx in enumerate(vx_seq) if ix not in filler_ixs]
    quantities = [q for ix, q in enumerate(quantities) if ix not in filler_ixs]
    rewards = (
        0
        if d in ini_and_dummy_vxs
        else max(d, sum(quantities[i] for vx in vx_seq if vx == i))
        for i, d in enumerate(demands)
    )
    cost = cost_calc_vrpp_core(
        vx_seq,
        dyn_costs,
        distance_mx,
        initial_vx,
        ini_and_dummy_vxs=ini_and_dummy_vxs,
        forbidden_val=forbidden_val,
    )
    reward = sum(rewards)
    return [cost, reward]


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
