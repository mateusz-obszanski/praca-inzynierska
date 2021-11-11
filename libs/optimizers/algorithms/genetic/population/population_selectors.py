from abc import ABC, abstractmethod
import itertools as it
import numpy as np

from ..population import Population
from ..operators.fixers import FixResult
from .....environment import Environment
from .....environment.cost.base import CostT


class PopulationSelector(ABC):
    @abstractmethod
    def select(
        self,
        new_generation: Population,
        old_generation: Population,
        environment: Environment,
        new_fix_results: list[FixResult],
        old_fix_results: list[FixResult],
        new_costs: list[CostT],
        old_costs: list[CostT],
        invalidity_weight: float,
        error_weight: float,
        cost_weight: float,
    ) -> tuple[Population, list[CostT], list[FixResult]]:
        ...


class PopulationSelectorProbabilistic(PopulationSelector):
    """
    Implements chromosome selection method based on costs and validity.
    """

    def select(
        self,
        new_generation: Population,
        old_generation: Population,
        environment: Environment,
        new_fix_results: list[FixResult],
        old_fix_results: list[FixResult],
        new_costs: list[CostT],
        old_costs: list[CostT],
        invalidity_weight: float,
        error_weight: float,
        cost_weight: float,
        n_best_to_bypass_grade: int = 0,
    ) -> tuple[Population, list[CostT], list[FixResult]]:
        """
        Assigns grades (probabilities) from 0 to 1 to each chromosome in joined
        old and new generations. Then draws the final population based on grades.

        Grade (not normalized):
            `(cost_weight + error_weight * no_of_errors + (invalidity_weight if is invalid)) * cost`

        `no_of_errors` is taken from fix results.

        Invalidity_weight discriminates against better sollutions (by cost) but not
        meeting other criteria.

        `error_weight` introduces gradation between invalid chromosomes.

        `validity_weight` must be smaller than `cost_weight`.

        `n_best_to_bypas_grade` lets some of te best solutions to be selected
        without grading.

        Assumes population size from length of old_generation.
        """

        assert invalidity_weight < cost_weight

        old_population = it.chain(new_generation, old_generation)
        costs = np.array(list(it.chain(new_costs, old_costs)), dtype=np.float64)

        sorted_old_population: Population

        sorted_old_population = [
            chromosome for _, chromosome in sorted(zip(costs, old_population))
        ]

        fix_results = list(it.chain(new_fix_results, old_fix_results))

        if n_best_to_bypass_grade:
            best_n = sorted_old_population[:n_best_to_bypass_grade]
            to_be_graded = sorted_old_population[n_best_to_bypass_grade:]
            costs_best_n = costs[:best_n]
            costs = costs[best_n:]
            fix_results = list(fix_results)
            best_n_fix_results = fix_results[:n_best_to_bypass_grade]
            fix_results = fix_results[n_best_to_bypass_grade:]
        else:
            best_n = []
            costs_best_n = []
            to_be_graded = sorted_old_population
            best_n_fix_results = []

        no_of_errors = np.array(
            [fix_result.no_of_errors for fix_result in fix_results], dtype=np.float64
        )
        grades = (
            cost_weight
            + error_weight * no_of_errors
            + invalidity_weight * bool(no_of_errors)
        ) * costs
        probabilities = grades / grades.sum()

        passed_ixs = np.random.choice(
            len(to_be_graded), p=probabilities, size=len(old_generation), replace=False
        )

        selected_population = best_n + [to_be_graded[ix] for ix in passed_ixs]
        selected_costs = costs_best_n + [costs[ix] for ix in passed_ixs]
        selected_fix_results = best_n_fix_results + [
            fix_results[ix] for ix in passed_ixs
        ]

        return selected_population, selected_costs, selected_fix_results
