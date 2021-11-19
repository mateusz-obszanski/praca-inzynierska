"""
Heuristic solution generators.
"""


from abc import ABC
import itertools as it
from operator import itemgetter
from ...utils.graph import graph_cycle_greedy_nn


from libs.solution.representation import SolutionRepresentationTSP
from . import (
    SolutionCreator,
    SolutionRepresentation,
    SolutionCreatorTSPSimple,
    EnvironmentTSPSimple,
)


class SolutionCreatorHeuristic(SolutionCreator, ABC):
    """
    Abstract base class.
    """


class SolutionCreatorTSPSimpleHeuristicNN(
    SolutionCreatorHeuristic, SolutionCreatorTSPSimple
):
    """
    Greedy nearest neighbour.
    """

    @staticmethod
    def create(
        environment: EnvironmentTSPSimple, initial_vx: int
    ) -> SolutionRepresentationTSP:
        """
        Performs greedy depth-first-search.
        """
        distance_mx = environment.cost
        return SolutionRepresentationTSP(graph_cycle_greedy_nn(distance_mx, initial_vx))
