from collections.abc import Iterable
from math import isfinite
from typing import Sequence, TypeVar, Union, overload, Literal, Protocol
import more_itertools as mit
import numpy as np

from libs.environment.cost_calculators import CostT
from libs.optimizers.algorithms.genetic.operators.fixers import FixStatus
from libs.utils.random import probabilities_by_value


# TODO tournament selection

Chromosome = np.ndarray
PairedChromosomes = tuple[Chromosome, Chromosome]
Population = Sequence[Chromosome]
Rng = TypeVar("Rng", bound=np.random.Generator)


class ParentSelector(Protocol):
    """
    Assumes that population size is even.
    """

    def __call__(
        self,
        population: Sequence[np.ndarray],
        costs: Sequence[CostT],
        rng: Rng,
    ) -> tuple[list[PairedChromosomes], list[CostT], Rng]:
        ...


class ParentSelectorDeprecated(Protocol):
    """
    Assumes that population size is even.
    """

    def __call__(
        self,
        population: Population,
        costs: Sequence[CostT],
        prev_fix_results: list[FixStatus],
        rng: Rng,
    ) -> tuple[list[PairedChromosomes], list[CostT], list[FixStatus], Rng]:
        """
        Sorts population by cost and groups into pairs (best, second best), ...
        Assumes that population size is even.
        """
        ...


def select_best_parents(
    population: Population,
    costs: Sequence[CostT],
    rng: Rng,
) -> tuple[list[PairedChromosomes], list[CostT], list[FixStatus], Rng]:
    """
    Sorts population by cost and groups into pairs (best, second best), ...
    Assumes that population size is even and costs positive.

    :returns tuple[list[tuple[Chromosome, Chromosome]], list[CostT]]: list
        of paired parents and list of input population costs.
    """

    sorted_population, costs = sort_population(population, costs)

    return (  # type: ignore
        list(mit.windowed(sorted_population, n=2, step=2)),
        costs,
        rng,
    )


def select_best_parents_with_probability(
    population: Population,
    costs: Sequence[CostT],
    rng: Rng,
) -> tuple[list[PairedChromosomes], list[CostT], Rng]:
    """
    Sorts popultaion by cost and groups into pairs. The best chromosomes have
    the highest probability of being chosen first, thus being paired with other
    good solutions.

    Assumes that population size is even and cost are positive.

    Transforming attractiveness i.e. by exp or taking it as 1 / costs does not
    have significant impact on mean difference of parent costs.
    """

    population_size = len(population)

    BIAS = 1e-3

    attractiveness = -np.array(costs, dtype=np.float64)
    attractiveness[~np.isfinite(attractiveness)] = 0
    # shift to [0, max-min]
    attractiveness += attractiveness.min()
    # scale to [0, 1]
    attractiveness += BIAS

    probabilities = probabilities_by_value(attractiveness)
    ixs_by_probability = rng.choice(
        population_size, p=probabilities, size=population_size, replace=False
    )

    paired_parents: list[PairedChromosomes] = [
        (population[i], population[j])  # type: ignore
        for i, j in mit.windowed(ixs_by_probability, n=2, step=2)
    ]

    parent_costs = [costs[ix] for ix in ixs_by_probability]

    return (paired_parents, parent_costs, rng)


@overload
def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
    return_indices: Literal[False],
) -> tuple[list[Chromosome], list[CostT]]:
    """
    Sorts population by cost in ascending order and returns calculated costs.
    """


@overload
def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
    return_indices: Literal[True],
) -> tuple[list[Chromosome], list[CostT], list[int]]:
    """
    Sorts population by cost in ascending order and returns calculated costs.
    """


@overload
def sort_population(
    population: Iterable[Chromosome],
    costs: Sequence[CostT],
) -> tuple[list[Chromosome], list[CostT]]:
    """
    Sorts population by cost in ascending order and returns calculated costs.
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
    Sorts population by cost in ascending order and returns calculated costs.
    """

    costs_len = len(costs)

    if return_indices:
        sorted_costs, sorted_population, sorted_ixs = mit.unzip(
            sorted(zip(costs, population, range(costs_len)), key=lambda x: (x[0], x[1]))
        )
        return list(sorted_costs), list(sorted_population), list(sorted_ixs)  # type: ignore

    sorted_costs, sorted_population = mit.unzip(sorted(zip(costs, population)))

    return list(sorted_population), list(sorted_costs)  # type: ignore
