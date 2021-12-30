"""
Random solution creators.
"""

from typing import TypeVar

import numpy as np

from libs.solution import SolutionTSP


Rng = TypeVar("Rng", bound=np.random.Generator)


def create_tsp_sol_rand(
    dist_mx: np.ndarray, initial_vx: int, rng: Rng
) -> tuple[SolutionTSP, Rng]:
    vx_n = dist_mx.shape[0]
    sol_mid = list(set(range(vx_n)) - {initial_vx})
    rng.shuffle(sol_mid)
    return [initial_vx] + sol_mid + [initial_vx], rng


def create_vrp_sol_rand(
    dist_mx: np.ndarray,
    initial_vx: int,
    rng: Rng,
    ini_and_dummy_vxs: set[int],
    regular_vx_division: bool = True,
) -> tuple[SolutionTSP, Rng]:
    """
    If `regular_vx_division`, assigns regular number of vertices to visit for
    each salesman. Assumes that `len(ini_and_dummy_vxs) == <salesmen no.>`.

    `dist_mx` must be extended.
    """
    vx_n = dist_mx.shape[0]
    salesmen_n = len(ini_and_dummy_vxs)
    if regular_vx_division:
        spacing = vx_n // salesmen_n
        # insert `salesmen_n - 1` dummy ixs
        dummy_vx_ixs = [i * spacing for i in range(1, salesmen_n)]
    else:
        # minimal spacing - 1
        dummy_vx_ixs = []
        # initial, at least one, ..., at least one, initial
        ix_pool = set(range(2, vx_n - 2))
        for _ in range(salesmen_n - 1):
            next_ix = rng.choice(ix_pool)
            ix_pool.difference_update((next_ix - 1, next_ix, next_ix + 1))
            dummy_vx_ixs.append(next_ix)
    if len(dummy_vx_ixs) + 1 != salesmen_n:
        raise TooLittleVerticesError(
            f"salesmen: {salesmen_n}, no. of all vertices incl. initial: {vx_n}",
            salesmen_n,
            vx_n,
        )
    vx_seq_mid = list(set(range(vx_n)) - ini_and_dummy_vxs)
    rng.shuffle(vx_seq_mid)
    vx_seq = [initial_vx] + vx_seq_mid + [initial_vx]
    for dix, dvx in zip(dummy_vx_ixs, ini_and_dummy_vxs - {initial_vx}):
        vx_seq.insert(dix, dvx)
    return vx_seq, rng


def create_irp_sol_rand(
    dist_mx: np.ndarray,
    initial_vx: int,
    rng: Rng,
    ini_and_dummy_vxs: set[int],
    vehicle_volume: float,
    regular_vx_division: bool = True,
) -> tuple[SolutionTSP, np.ndarray, Rng]:
    """
    If `regular_vx_division`, assigns regular number of vertices to visit for
    each salesman. Assumes that `len(ini_and_dummy_vxs) == <salesmen no.>`.

    Returns vertex sequence, quantity to leave list and provided random number
    generator.
    """
    vx_seq, rng = create_vrp_sol_rand(
        dist_mx,
        initial_vx,
        rng,
        ini_and_dummy_vxs,
        regular_vx_division=regular_vx_division,
    )
    quantities: np.ndarray = rng.random(size=len(vx_seq))
    dummy_ixs: np.ndarray = np.fromiter(
        (vx in ini_and_dummy_vxs for vx in vx_seq), dtype=np.bool8
    )
    quantities[dummy_ixs] = 0.0
    quantities *= vehicle_volume / quantities.sum()
    return vx_seq, quantities, rng


class TooLittleVerticesError(Exception):
    """
    Raised by random solution generator if there is insufficient number
    of vertices to distribute between salesmen.
    """
