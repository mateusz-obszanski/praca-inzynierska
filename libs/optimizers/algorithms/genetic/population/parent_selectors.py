from collections.abc import Iterable
from typing import Sequence, TypeVar, Union, overload, Literal, Protocol
import more_itertools as mit
import numpy as np

from . import Population, Chromosome
from ..operators.fixers import FixResult
from .....environment.cost_calculators import CostT
from .....utils.random import probabilities_by_value


PairedChromosomes = tuple[Chromosome, Chromosome]
Rng = TypeVar("Rng", bound=np.random.Generator)


class ParentSelector(Protocol):
    """
    Assumes that population size is even.
    """

    def __call__(
        self,
        population: Population,
        costs: Sequence[CostT],
        prev_fix_results: list[FixResult],
        rng: Rng,
    ) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult], Rng]:
        """
        Sorts population by cost and groups into pairs (best, second best), ...
        Assumes that population size is even.
        """


def select_best_parents(
    population: Population,
    costs: Sequence[CostT],
    prev_fix_results: list[FixResult],
) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult]]:
    """
    Sorts population by cost and groups into pairs (best, second best), ...
    Assumes that population size is even and costs positive.

    :returns tuple[list[tuple[Chromosome, Chromosome]], list[CostT]]: list
        of paired parents and list of input population costs.
    """

    sorted_population, costs, fix_results_ixs = sort_population(
        population, costs, return_indices=True
    )

    return (  # type: ignore
        list(mit.windowed(sorted_population, n=2, step=2)),
        costs,
        [prev_fix_results[ix] for ix in fix_results_ixs],
    )


def select_best_parents_with_probability(
    population: Population,
    costs: Sequence[CostT],
    prev_fix_results: list[FixResult],
    rng: Rng,
) -> tuple[list[PairedChromosomes], list[CostT], list[FixResult], Rng]:
    """
    Sorts popultaion by cost and groups into pairs. The best chromosomes have
    the highest probability of being chosen first, thus being paired with other
    good solutions.

    Assumes that population size is even and cost are positive.
    """

    population_size = len(population)

    probabilities = probabilities_by_value(costs)
    ixs_by_probability = rng.choice(
        range(population_size), p=probabilities, size=population_size, replace=False
    )

    paired_parents: list[PairedChromosomes] = [
        (population[i], population[j])  # type: ignore
        for i, j in mit.windowed(ixs_by_probability, n=2, step=2)
    ]

    parent_costs = [costs[ix] for ix in ixs_by_probability]
    parent_fix_results = [prev_fix_results[ix] for ix in ixs_by_probability]

    return (paired_parents, parent_costs, parent_fix_results, rng)


@overload
def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
    return_indices: Literal[False],
) -> tuple[list[Chromosome], list[CostT]]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """


@overload
def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
    return_indices: Literal[True],
) -> tuple[list[Chromosome], list[CostT], list[int]]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """


def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
    return_indices: bool = False,
) -> Union[
    tuple[list[Chromosome], list[CostT]],
    tuple[list[Chromosome], list[CostT], list[int]],
]:
    """
    Sorts population by cost in descending order and returns calculated costs.
    """

    if return_indices:
        sorted_costs, sorted_population, sorted_ixs = mit.unzip(
            sorted(
                zip(costs, population, range(len(costs))), key=lambda x: (x[0], x[1])
            )
        )
        return list(sorted_costs), list(sorted_population), list(sorted_ixs)  # type: ignore

    sorted_costs, sorted_population = mit.unzip(sorted(zip(costs, population)))

    return list(sorted_population), list(sorted_costs)  # type: ignore
