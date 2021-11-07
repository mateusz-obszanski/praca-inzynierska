"""
Cost functions for Travelling Salesman Problem.
"""


from typing import Generator
from . import SolutionRepresentationTSP, EnvironmentTSPSimple
from .base import CostFunctor, CostT


# TODO for dynamic environments, implement generators/iterators that accumulate
# costs and maybe yield special event flags/info


class TSPCostFunctor(CostFunctor):
    ...


class TSPCostFuntorSimple(TSPCostFunctor):
    @staticmethod
    def calculate(
        solution: SolutionRepresentationTSP, environment: EnvironmentTSPSimple
    ) -> Generator[CostT, None, None]:
        solution_representation = solution.representation
        distance_mx = environment.cost

        current_vx = solution_representation[0]

        for vx in solution_representation[1:]:
            yield distance_mx[current_vx, vx]
            current_vx = vx
