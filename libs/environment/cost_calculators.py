from typing import Generator

import numpy as np
import more_itertools as mit

from ..solution import SolutionTSP


CostT = float
CostGen = Generator[CostT, None, None]


def calculate_total_cost(cost_generator: CostGen) -> CostT:
    return sum(cost_generator)


def cost_tsp_gen(
    solution: SolutionTSP, cost_mx: np.ndarray, initial_ix: int = 0
) -> CostGen:
    yield cost_mx[initial_ix, solution[0]]

    yield from (
        cost_mx[current_vx, next_vx]
        for current_vx, next_vx in mit.windowed(solution, n=2)
    )
