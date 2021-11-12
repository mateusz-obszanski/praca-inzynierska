"""
New population generators.
"""


from typing import TypedDict
from collections.abc import Iterator
import more_itertools as mit

from . import Population
from .population_selectors import PopulationSelector
from .parent_selectors import ParentSelector
from ..chromosomes import Chromosome
from ..operators.mutations import Mutator
from ..operators.crossovers import Crossover
from ..operators.fixers import ChromosomeFixer, FixResult, FixStatus
from .....environment import Environment
from .....environment.cost import CostCalculator
from .....environment.cost.base import CostT
from .....solution.representation import SolutionRepresentationTSP


# TODO used concrete SolutionRepresentationTSP - make conversions between chromosomes and representations
# or use idfferent classes


class PopulationGenerationData(TypedDict):
    parent_costs: list[CostT]
    new_costs: list[CostT]
    did_mutate: dict[Mutator, list[bool]]
    fix_results: list[FixResult]


class PopulationGenerator:
    @staticmethod
    def generate(
        old_population: Population,
        environment: Environment,
        cost_calculator: CostCalculator,
        parent_selector: ParentSelector,
        mutators: list[Mutator],
        crossover: Crossover,
        fixer: ChromosomeFixer,
        population_selector: PopulationSelector,
        invalidity_weight: float,
        error_weight: float,
        cost_weight: float,
    ) -> tuple[Population, PopulationGenerationData]:
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

        population_to_be_mutated: Iterator[Chromosome] = iter(old_population)
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
                fixer.number_of_errors(chromosome, environment)
                for chromosome in mutated_population
            )
        ]

        parents, parent_costs, parent_fix_results = parent_selector.select(
            mutated_population,
            environment,
            cost_calculator,
            mutated_population_fix_results,
        )

        new_generation = list(mit.flatten(
            crossover.execute(parent1, parent2) for parent1, parent2 in parents
        ))

        new_generation_fixed: list[Chromosome]
        new_generation_fix_results: list[FixResult]

        new_generation_fixed, new_generation_fix_results = map(list, mit.unzip(  # type: ignore
            fixer.fix(chromosome, environment) for chromosome in new_generation
        ))

        old_generation = list(mit.flatten(parents))
        old_generation_costs = parent_costs
        old_generation_fix_results = parent_fix_results

        new_generation_costs = [
            cost_calculator.calculate_total(SolutionRepresentationTSP(chromosome.sequence), environment)[0]
            for chromosome in new_generation_fixed
        ]

        (
            new_population,
            new_population_costs,
            new_population_fix_results,
        ) = population_selector.select(
            new_generation_fixed,
            old_generation,
            environment,
            new_generation_fix_results,
            old_generation_fix_results,
            new_generation_costs,
            old_generation_costs,
            invalidity_weight,
            error_weight,
            cost_weight,
        )

        generation_data: PopulationGenerationData = {
            "parent_costs": parent_costs,
            "new_costs": new_population_costs,
            "did_mutate": mutation_data,
            "fix_results": new_population_fix_results,
        }

        return new_population, generation_data
