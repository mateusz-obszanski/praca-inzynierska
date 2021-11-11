from abc import ABC, abstractmethod
from collections.abc import Iterable
import more_itertools as mit
import numpy.random as np_rnd

from . import Population
from ..chromosomes import Chromosome
from .....solution.representation import SolutionRepresentation
from .....environment import Environment
from .....environment.cost import CostCalculator, CostT
from .....utils.random import probabilities_by_value


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
    ) -> tuple[list[PairedChromosomes], list[CostT]]:
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
    ) -> tuple[list[PairedChromosomes], list[CostT]]:
        """
        Sorts population by cost and groups into pairs (best, second best), ...
        Assumes that population size is even and costs positive.

        :returns tuple[list[tuple[Chromosome, Chromosome]], list[CostT]]: list
            of paired parents and list of input population costs.
        """

        sorted_population, costs = sort_population(
            population, environment, cost_calculator
        )

        return list(mit.windowed(sorted_population, n=2, step=2)), costs  # type: ignore


class ParentSelectorElitistRandomized:
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
    ) -> tuple[list[PairedChromosomes], list[CostT]]:
        """
        Sorts popultaion by cost and groups into pairs. The best chromosomes have
        the highest probability of being chosen first, thus being paired with other
        good solutions.

        Assumes that population size is even and cost are positive.
        """

        population_size = len(population)

        costs = [
            cost_calculator.calculate_total(chromosome.sequence, environment)[0]
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

        return paired_parents, costs


def sort_population(
    population: Iterable[Chromosome],
    environment: Environment,
    cost_calculator: CostCalculator,
) -> tuple[list[Chromosome], list[CostT]]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """

    costs = [
        cost_calculator.calculate_total(
            SolutionRepresentation(chromosome.sequence), environment
        )[0]
        for chromosome in population
    ]

    sorted_costs, sorted_population = mit.unzip(sorted(zip(costs, population)))

    return sorted_population, sorted_costs  # type: ignore
