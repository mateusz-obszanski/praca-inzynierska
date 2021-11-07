"""
Cost functions for Travelling Salesman Problem.
"""


from abc import ABC
from typing import Generator
from . import SolutionRepresentationTSP, EnvironmentTSPSimple
from .base import CostCalculator, CostT


# TODO for dynamic environments, implement generators/iterators that accumulate
# costs and maybe yield special event flags/info


class TSPCostCalculator(CostCalculator, ABC):
    """
    Abstract base class.
    """


class TSPCostCalculatorSimple(TSPCostCalculator):
    @staticmethod
    def _stepper(
        solution: SolutionRepresentationTSP, environment: EnvironmentTSPSimple
    ) -> Generator[CostT, None, None]:
        solution_representation = solution.representation
        distance_mx = environment.cost

        current_vx = solution_representation[0]

        for vx in solution_representation[1:]:
            yield distance_mx[current_vx, vx]
            current_vx = vx
