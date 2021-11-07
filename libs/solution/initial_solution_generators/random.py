"""
Random solution generators.
"""


from abc import ABC
import random


from . import (
    SolutionCreator,
    SolutionRepresentation,
    SolutionRepresentationTSP,
    SolutionCreatorTSPSimple,
    SolutionCreatorTSP,
    EnvironmentTSPSimple,
)


class SolutionCreatorRandom(SolutionCreator, ABC):
    """
    Abstract base class.
    """


class SolutionCreatorTSPSimpleRandom(SolutionCreatorRandom, SolutionCreatorTSPSimple):
    @staticmethod
    def create(
        environment: EnvironmentTSPSimple, initial_vx: int
    ) -> SolutionRepresentationTSP:
        distance_mx = environment.cost
        vx_n, _ = distance_mx.shape
        random_solution_tail = list(set(range(vx_n)) - {initial_vx})
        random.shuffle(random_solution_tail)
        return SolutionRepresentationTSP([initial_vx] + random_solution_tail)
