"""
Heuristic solution creators.
"""

import numpy as np

from ...utils.graph import graph_cycle_greedy_nn


def create_tsp_solution_nearest_neighbour(
    cost_mx: np.ndarray, initial_vx: int, forbidden_val: float = -1
):
    # FIXME vvv does not return a cycle - missing initial_vx at the end
    # for grid it doesn't matter
    return graph_cycle_greedy_nn(cost_mx, initial_vx, forbidden_val)
