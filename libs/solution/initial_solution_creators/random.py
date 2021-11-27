"""
Random solution creators.
"""


import random

import numpy as np

from .. import SolutionTSP


def create_tsp_solution_random(
    cost_mx: np.ndarray, initial_vx: int, rng: np.random.Generator
) -> SolutionTSP:
    vx_n = cost_mx.shape[0]
    solution_tail = list(set(range(vx_n)) - {initial_vx})
    rng.shuffle(solution_tail)
    return [initial_vx] + solution_tail
