from typing import Generator, Protocol

import numpy as np
import more_itertools as mit

from libs.solution import SolutionTSP


CostT = float
CostGen = Generator[CostT, None, None]


class CostGenCreator(Protocol):
    """
    Assumes that solution does not start from initial vx.
    """

    def __call__(self, solution, cost_mx: np.ndarray, initial_ix: int = 0) -> CostGen:
        """
        Assumes that solution does not start from initial vx.
        """


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
