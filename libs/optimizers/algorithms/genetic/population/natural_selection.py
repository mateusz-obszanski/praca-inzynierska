import itertools as it
import numpy as np
from operator import itemgetter
from typing import Protocol, TypeVar
from collections.abc import Sequence

import more_itertools as mit

from libs.environment.cost_calculators import CostT


Rng = TypeVar("Rng", bound=np.random.Generator)
T = TypeVar("T")
Chromosome = np.ndarray
Population = list[Chromosome]


class NaturalSelector(Protocol):
    def __call__(
        self,
        parents: Population,
        offspring: Population,
        parent_costs: list[CostT],
        offspring_costs: list[CostT],
        rng: Rng,
        *args,
        **kwargs,
    ) -> tuple[Population, list[CostT], Rng]:
        ...


def select_population_with_probability(
    parents: Population,
    offspring: Population,
    parent_costs: list[CostT],
    offspring_costs: list[CostT],
    rng: Rng,
    n_best_bypass: int = 0,
) -> tuple[Population, list[CostT], Rng]:
    """
    Implements probabilistic chromosome selection method based on costs
    and validity. Assigns grades (probabilities) from 0 to 1 to each chromosome
    in joined old and new generations. Then draws the final population based
    on attractiveness.

    `n_best_bypass` lets some of te best solutions to be selected without grading.

    Deduces population size from length of the old_generation. Assumes that
    parents represent valid solutions.
    """

    assert n_best_bypass < len(parents)

    BIAS = 1e-2

    old_gen = it.chain(offspring, parents)
    old_costs = np.array(
        list(it.chain(offspring_costs, parent_costs)), dtype=np.float64
    )

    sorted_old_population: Population

    sorted_old_population = [
        chromosome
        for _, chromosome in sorted(zip(old_costs, old_gen), key=itemgetter(0))
    ]

    if n_best_bypass:
        best_n = sorted_old_population[:n_best_bypass]
        costs_best_n = old_costs[:best_n]
        to_be_graded = sorted_old_population[n_best_bypass:]
        old_costs = old_costs[best_n:]
    else:
        best_n = []
        costs_best_n = []
        to_be_graded = sorted_old_population

    # shift on y axis to 0 and add BIAS
    grades = -old_costs + old_costs.max() + BIAS

    probabilities = grades / grades.sum()

    to_be_graded_len = len(to_be_graded)

    passed_ixs = rng.choice(
        to_be_graded_len, p=probabilities, size=to_be_graded_len, replace=False
    )

    selected_population = best_n + [to_be_graded[ix] for ix in passed_ixs]
    selected_costs = costs_best_n + [old_costs[ix] for ix in passed_ixs]

    return selected_population, selected_costs, rng


def replace_invalid_offspring(
    parents: Sequence[tuple[np.ndarray, np.ndarray]],
    offspring: list[np.ndarray],
    parent_costs: Sequence[CostT],
    offspring_fix_statuses: Sequence,
) -> list[np.ndarray]:
    assert 2 * len(parents) == len(parent_costs)
    assert 2 * len(parents) == len(offspring)
    assert len(offspring) == len(offspring_fix_statuses)
    fail_ixs = [i for i, success in offspring_fix_statuses if not success]
    # fail pairs that have the same parents
    fail_ix_pairs: tuple[tuple[int, int]] = tuple(  # type: ignore
        (i1, i2)
        for i1, i2 in mit.windowed(fail_ixs, n=2)
        if i1 % 2 == 0 and i1 == i2 - 1  # type: ignore
    )
    for i1, i2 in fail_ix_pairs:
        p1, p2 = parents[i1 // 2]
        i2p1 = i2 + 1
        offspring[i1:i2p1] = p1.copy(), p2.copy()
        fail_ixs[i1:i2p1] = -1, -1
    for ix in (i for i in fail_ixs if i != -1):
        p1c, p2c = parent_costs[ix : ix + 2]
        if p1c < p2c:
            p = parents[ix // 2][0]
        else:
            p = parents[ix // 2][1]
        offspring[ix] = p.copy()

    return offspring
