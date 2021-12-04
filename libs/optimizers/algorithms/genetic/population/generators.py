"""
New population generators.
"""


from dataclasses import dataclass
from collections.abc import Iterator
from typing import Sequence, TypeVar
import more_itertools as mit

import numpy as np

from .chromosomes import ChromosomeTSP
from .population_selectors import PopulationSelector
from .parent_selectors import ParentSelector
from ..operators.mutations import Mutator
from ..operators.crossovers import Crossover
from ..operators.fixers import Fixer, FixResult, FixStatus, check_transitions_deprecated
from .....environment.cost_calculators import (
    CostGenCreator,
    CostT,
    calculate_total_cost,
)


Chromosome = TypeVar("Chromosome", bound=list[int])
Rng = TypeVar("Rng", bound=np.random.Generator)


@dataclass(frozen=True)
class PopulationGenerationData:
    parent_costs: list[CostT]
    new_costs: list[CostT]
    did_mutate: dict[Mutator, list[bool]]
    fix_results: list[FixResult]


def generate_population(
    old_population: Sequence[ChromosomeTSP],
    cost_mx: np.ndarray,
    cost_gen_creator: CostGenCreator,
    parent_selector: ParentSelector,
    mutators: list[Mutator],
    crossover: Crossover,
    fixer: Fixer,
    population_selector: PopulationSelector,
    invalidity_weight: float,
    error_weight: float,
    cost_weight: float,
    rng: Rng,
) -> tuple[list[ChromosomeTSP], PopulationGenerationData, Rng]:
    """
    Generates new population from the old one using provided mutators,
    crossovers and fixers. If certain offspring cannot be fixed, takes
    better mutated parent instead (or two if both children are unrecoverable).

    New generation is selected by fitness to be the same size as the old one.

    Assumes that population size is even.

    `old_population_prev_fix_results` for the first time can be set to all
    successes or generated by mapping fixer onto old_population.

    Returns new population
    """

    population_size = len(old_population)

    assert population_size % 2 == 0

    population_to_be_mutated = iter(old_population)
    did_mutate: list[bool] = []
    mutation_data: dict[Mutator, list[bool]] = {}

    for mutator in mutators:
        population_to_be_mutated, did_mutate = mit.unzip(
            mutator.mutate(chromosome) for chromosome in population_to_be_mutated  # type: ignore
        )

        mutation_data[mutator] = list(did_mutate)

    mutated_population = list(population_to_be_mutated)

    # only for parent_selector, only cares for number of errors
    mutated_population_fix_results = [
        FixResult(FixStatus.SUCCESS)
        if not no_of_errors
        else FixResult(FixStatus.FAILURE, no_of_errors)
        for no_of_errors in (
            sum(
                not valid_transition
                for valid_transition in check_transitions_deprecated(
                    chromosome, cost_mx
                )
            )
            for chromosome in mutated_population
        )
    ]

    mutated_costs = [
        calculate_total_cost(cost_gen_creator(chromosome, cost_mx))
        for chromosome in mutated_population
    ]

    parents, parent_costs, parent_fix_results, rng = parent_selector(
        mutated_population,  # type: ignore
        mutated_costs,
        mutated_population_fix_results,
        rng,
    )

    new_generation = list(
        mit.flatten(
            crossover(parent1, parent2, rng) for parent1, parent2 in parents  # type: ignore
        )
    )

    new_generation_fixed: list[ChromosomeTSP]
    new_generation_fix_results: list[FixResult]

    new_generation_fixed, new_generation_fix_results = map(
        list,
        mit.unzip(  # type: ignore
            fixer(chromosome, cost_mx) for chromosome in new_generation
        ),
    )

    old_generation = list(mit.flatten(parents))
    old_generation_costs = parent_costs
    old_generation_fix_results = parent_fix_results

    new_generation_costs = [
        calculate_total_cost(cost_gen_creator(chromosome, cost_mx))
        for chromosome in new_generation_fixed
    ]

    (
        new_population,
        new_population_costs,
        new_population_fix_results,
        rng,
    ) = population_selector(
        new_generation_fixed,  # type: ignore
        old_generation,  # type: ignore
        new_generation_fix_results,
        old_generation_fix_results,
        new_generation_costs,
        old_generation_costs,
        invalidity_weight,
        error_weight,
        cost_weight,
        rng,
    )

    generation_data = PopulationGenerationData(
        parent_costs,
        new_population_costs,
        mutation_data,
        new_population_fix_results,
    )

    return new_population, generation_data, rng  # type: ignore
