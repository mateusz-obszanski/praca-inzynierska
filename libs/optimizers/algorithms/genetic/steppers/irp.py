import itertools as it
from typing import Any

import numpy as np

from libs.environment.cost_calculators import DistMx, DynCosts, cost_calc_irp
from libs.environment.utils import check_transitions
from libs.optimizers.algorithms.genetic.population.parent_selectors import (
    select_best_parents_with_probability,
)
from libs.optimizers.algorithms.genetic.operators.mutations import Mutator
from libs.optimizers.algorithms.genetic.operators.crossovers import CrossoverNDArray
from libs.optimizers.algorithms.genetic.operators.fixers import fix_irp
from libs.optimizers.algorithms.genetic.population.natural_selection import (
    replace_invalid_offspring,
    select_population_with_probability,
)
from libs.utils.matrix import extend_cost_mx
from .utils import ExpStepper, NextGenData, apply_mutators_irp
from .exceptions import InitialPopInvalidError


Chromosome = np.ndarray
Population = list[Chromosome]
Rng = np.random.Generator


def genetic_stepper_irp(
    population: Population,
    dyn_costs: DynCosts,
    dist_mx: DistMx,
    crossover: CrossoverNDArray,
    mutators: list[Mutator],
    mut_ps: dict[Mutator, float],
    crossover_kwargs: dict[str, Any],
    mut_kwargs: dict[Mutator, dict[str, Any]],
    initial_vx: int,
    fix_max_add_iters: int,
    fix_max_retries: int,
    rng: Rng,
    salesmen_n: int,
    demands: tuple[float, ...],
    fillval: int,
    salesman_capacity: float,
) -> ExpStepper:
    """
    For the first time yields initial population, its data and rng

    If `natural_selector` is `None`, then offspring becomes the next generation.
    """

    # [(mx, exp_t)] -> (mx, exp_t) -> mx -> mx[0, 0]
    forbidden_val = dyn_costs[0][0][0, 0]
    pop_lists: tuple[list[int], ...]
    quantity_lists: tuple[list[float], ...]
    pop_lists, quantity_lists = zip(
        *((c[:, 0].tolist(), c[:, 1].tolist()) for c in population)
    )
    copy_n = salesmen_n - 1
    ext_dist_mx = extend_cost_mx(dist_mx, copy_n, to_copy_ix=initial_vx)
    ext_dyn_costs = tuple(
        (extend_cost_mx(mx, copy_n, to_copy_ix=initial_vx), t) for mx, t in dyn_costs
    )
    fix_statuses = [check_transitions(c, ext_dist_mx, forbidden_val) for c in pop_lists]
    if any(not valid for valid in fix_statuses):
        inv_ixs = [i for i, success in enumerate(fix_statuses) if not success]
        raise InitialPopInvalidError("some initial individuals were not valid", inv_ixs)
    ini_and_dummy_vxs = {*range(salesmen_n)}
    cost_vecs = [
        cost_calc_irp(
            c,
            ext_dyn_costs,
            ext_dist_mx,
            initial_vx,
            forbidden_val,
            ini_and_dummy_vxs,
            demands,
            fillval,
            salesmen_n,
            qs,
            salesman_capacity,
        )
        for c, qs in zip(pop_lists, quantity_lists)
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
            apply_mutators_irp(
                c[:, 0], mutators, mut_ps, mut_kwargs, rng, quantities=c[:, 1]
            )[0]
            for c in offspring
        ]
        fix_results = [
            fix_irp(
                c.astype(int).tolist(),
                ext_dist_mx,
                initial_vx,
                forbidden_val,
                fix_max_add_iters,
                fix_max_retries,
                fillval,
                default_quantity=0,
                quantities=qs.tolist(),
                capacity=salesman_capacity,
            )
            for c, qs in mutated_offspring
        ]
        fixed_offspring = [np.stack((r[0], r[1]), axis=1) for r in fix_results]
        fix_statuses = [r[2] for r in fix_results]
        checked_offspring = replace_invalid_offspring(
            parents, fixed_offspring, parent_costs, fix_statuses
        )
        offspring_cost_vecs = [
            cost_calc_irp(
                c[:, 0].astype(int).tolist(),
                ext_dyn_costs,
                ext_dist_mx,
                initial_vx,
                forbidden_val,
                ini_and_dummy_vxs,
                demands,
                fillval,
                salesmen_n,
                quantities=c[:, 1].tolist(),
                capacity=salesman_capacity,
            )
            for c in checked_offspring
        ]
        # cost - reward
        offspring_costs = [v[0] - v[1] for v in offspring_cost_vecs]
        flattened_parents = list(it.chain.from_iterable(parents))
        next_gen, next_gen_costs, rng = select_population_with_probability(
            flattened_parents, checked_offspring, parent_costs, offspring_costs, rng
        )
        no_of_failures = sum(not success for success in fix_statuses)
        generation_data = NextGenData(
            next_gen_costs, no_of_failures, mut_ps, cross_inv_p
        )
        yield next_gen, generation_data, rng  # type: ignore

        population = next_gen
        costs = next_gen_costs
