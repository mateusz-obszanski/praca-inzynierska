from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union, overload, Literal
import more_itertools as mit
import numpy.random as np_rnd

from . import Population
from ..chromosomes import Chromosome
from ..operators.fixers import FixResult
from .....solution.representation import SolutionRepresentation
from .....environment import Environment
from .....environment.cost import CostCalculator, CostT
from .....utils.random import probabilities_by_value
from .....solution.representation import SolutionRepresentationTSP


PairedChromosomes = tuple[Chromosome, Chromosome]


class ParentSelector(ABC):
    """
    Abstract base class. Assumes that population size is even.
    """

    @abstractmethod
    def select(
        self,
        population: Population,
        environment: Environment,
        cost_calculator: CostCalculator,
        prev_fix_results: list[FixResult],
    ) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult]]:
        """
        Assumes that population size is even.
        """


class ParentSelectorElitist(ParentSelector):
    """
    Sorts population by cost and groups into pairs (best, second best), ...
    Assumes that population size is even.
    """

    def select(
        self,
        population: Population,
        environment: Environment,
        cost_calculator: CostCalculator,
        prev_fix_results: list[FixResult],
    ) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult]]:
        """
        Sorts population by cost and groups into pairs (best, second best), ...
        Assumes that population size is even and costs positive.

        :returns tuple[list[tuple[Chromosome, Chromosome]], list[CostT]]: list
            of paired parents and list of input population costs.
        """

        sorted_population, costs, fix_results_ixs = sort_population(
            population, environment, cost_calculator, return_indices=True
        )

        return (  # type: ignore
            list(mit.windowed(sorted_population, n=2, step=2)),
            costs,
            [prev_fix_results[ix] for ix in fix_results_ixs],
        )


class ParentSelectorElitistRandomized(ParentSelector):
    """
    Sorts popultaion by cost and groups into pairs. The best chromosomes have
    the highest probability of being chosen first, thus being paired with other
    good solutions.
    """

    def select(
        self,
        population: Population,
        environment: Environment,
        cost_calculator: CostCalculator,
        prev_fix_results: list[FixResult],
    ) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult]]:
        """
        Sorts popultaion by cost and groups into pairs. The best chromosomes have
        the highest probability of being chosen first, thus being paired with other
        good solutions.

        Assumes that population size is even and cost are positive.
        """

        population_size = len(population)

        costs = [
            cost_calculator.calculate_total(
                SolutionRepresentationTSP(chromosome.sequence), environment
            )[0]
            for chromosome in population
        ]

        probabilities = probabilities_by_value(costs)
        ixs_by_probability = np_rnd.choice(
            range(population_size), p=probabilities, size=population_size, replace=False
        )

        paired_parents: list[PairedChromosomes] = [
            (population[i], population[j])  # type: ignore
            for i, j in mit.windowed(ixs_by_probability, n=2, step=2)
        ]

        return (
            paired_parents,
            costs,
            [prev_fix_results[ix] for ix in ixs_by_probability],
        )


@overload
def sort_population(
    population: Iterable[Chromosome],
    environment: Environment,
    cost_calculator: CostCalculator,
    return_indices: Literal[False],
) -> tuple[list[Chromosome], list[CostT]]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """


@overload
def sort_population(
    population: Iterable[Chromosome],
    environment: Environment,
    cost_calculator: CostCalculator,
    return_indices: Literal[True],
) -> tuple[list[Chromosome], list[CostT], list[int]]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """


def sort_population(
    population: Iterable[Chromosome],
    environment: Environment,
    cost_calculator: CostCalculator,
    return_indices: bool = False,
) -> Union[
    tuple[list[Chromosome], list[CostT]],
    tuple[list[Chromosome], list[CostT], list[int]],
]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """

    costs = [
        cost_calculator.calculate_total(
            SolutionRepresentation(chromosome.sequence), environment
        )[0]
        for chromosome in population
    ]

    if return_indices:
        sorted_costs, sorted_population, sorted_ixs = mit.unzip(
            sorted(
                zip(costs, population, range(len(costs))), key=lambda x: (x[0], x[1])
            )
        )
        return list(sorted_costs), list(sorted_population), list(sorted_ixs)  # type: ignore

    sorted_costs, sorted_population = mit.unzip(sorted(zip(costs, population)))

    return list(sorted_population), list(sorted_costs)  # type: ignore
