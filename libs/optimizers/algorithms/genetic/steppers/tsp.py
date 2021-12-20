import itertools as it
from typing import Any

import numpy as np

from libs.environment.cost_calculators import DistMx, DynCosts, cost_calc_tsp
from libs.optimizers.algorithms.genetic.population.parent_selectors import (
    select_best_parents_with_probability,
)
from libs.optimizers.algorithms.genetic.operators.mutations import Mutator
from libs.optimizers.algorithms.genetic.operators.crossovers import CrossoverNDArray
from libs.optimizers.algorithms.genetic.operators.fixers import (
    check_chromosome_tsp,
    fix_tsp,
)
from libs.optimizers.algorithms.genetic.population.natural_selection import (
    replace_invalid_offspring,
    select_population_with_probability,
)
from .utils import ExpStepper, NextGenData, apply_mutators
from .exceptions import InitialPopInvalidError


Chromosome = np.ndarray
Population = list[Chromosome]
Rng = np.random.Generator


def genetic_stepper_tsp(
    population: Population,
    dyn_costs: DynCosts,
    dist_mx: DistMx,
    crossover: CrossoverNDArray,
    mutators: list[Mutator],
    mut_ps: dict[Mutator, float],
    crossover_kwargs: dict[str, Any],
    mut_kwargs: list[dict[str, Any]],
    initial_vx: int,
    fix_max_add_iters: int,
    fix_max_retries: int,
    rng: Rng,
) -> ExpStepper:
    """
    For the first time yields initial population, its data and rng

    If `natural_selector` is `None`, then offspring becomes the next generation.
    """

    # [(mx, exp_t)] -> (mx, exp_t) -> mx -> mx[0, 0]
    forbidden_val = dyn_costs[0][0][0, 0]
    pop_lists = tuple(c.tolist() for c in population)
    fix_statuses = [
        check_chromosome_tsp(c, dist_mx, initial_vx, forbidden_val) for c in pop_lists
    ]
    if any(not valid for valid in fix_statuses):
        inv_ixs = [i for i, success in enumerate(fix_statuses) if not success]
        raise InitialPopInvalidError("some initial individuals were not valid", inv_ixs)
    cost_vecs = [
        cost_calc_tsp(c, dyn_costs, dist_mx, initial_vx, forbidden_val)
        for c in pop_lists
    ]
    costs = [v[0] for v in cost_vecs]
    cross_inv_p = crossover_kwargs.get("inversion_p")
    no_of_failures = sum(not success for success in fix_statuses)
    initial_data = NextGenData(costs, no_of_failures, mut_ps, cross_inv_p)
    yield population, initial_data, rng  # type: ignore
    while True:
        parents, parent_costs, rng = select_best_parents_with_probability(
            population, costs, rng
        )
        offspring = list(
            it.chain.from_iterable(
                crossover(p1, p2, rng, **crossover_kwargs)[:-1] for p1, p2 in parents
            )
        )
        mutated_offspring = [
            apply_mutators(c, mutators, mut_ps, mut_kwargs, rng)[0] for c in offspring
        ]
        fix_results = [
            fix_tsp(
                c.tolist(),
                dist_mx,
                rng,
                forbidden_val,
                fix_max_add_iters,
                fix_max_retries,
            )
            for c in mutated_offspring
        ]
        fixed_offspring = [np.array(r[0], dtype=np.int64) for r in fix_results]
        fix_statuses = [r[1] for r in fix_results]
        checked_offspring = replace_invalid_offspring(
            parents, fixed_offspring, parent_costs, fix_statuses
        )
        offspring_cost_vecs = [
            cost_calc_tsp(
                c.tolist(),
                dyn_costs,
                dist_mx,
                initial_vx,
                forbidden_val=forbidden_val,
            )
            for c in checked_offspring
        ]
        offspring_costs = [v[0] for v in offspring_cost_vecs]
        flattened_parents = list(it.chain.from_iterable(parents))
        next_gen, next_gen_costs, rng = select_population_with_probability(
            flattened_parents, checked_offspring, parent_costs, offspring_costs, rng
        )
        no_of_failures = sum(not success for success in fix_statuses)
        generation_data = NextGenData(
            next_gen_costs, no_of_failures, mut_ps, cross_inv_p
        )
        yield next_gen, generation_data, rng  # type: ignore

        parents = next_gen
        costs = next_gen_costs
