from functools import partial
from pathlib import Path
from time import time

from more_itertools import take
import numpy as np
from bin.utils import ExpEndReason, process_generation_data, write_results

from libs.schemas.exp_funcs_map import ExperimentType
from libs.data_loading import get_experiment_config
from libs.optimizers.algorithms.genetic.steppers.tsp import genetic_stepper_tsp
from libs.utils.iteration import iterate_dataclass


# TODO TSP experiment run
# TODO IRP run


Chromosome = np.ndarray
Population = list[Chromosome]


def experiment(exp_t: str, exp_conf_path: str, results_path: str):
    # TODO other than TSP
    # TODO read fields from config file
    # TODO export as CLI
    _results_path = Path(results_path)
    del results_path
    _exp_t = ExperimentType[exp_t.upper()]
    del exp_t
    exp_config = get_experiment_config(_exp_t, exp_conf_path)
    population: Population = [
        individual["vx_seq"] for individual in exp_config.initial_population
    ]
    rng = np.random.default_rng(exp_config.rng_seed)
    stepper = genetic_stepper_tsp(
        population=population,
        dyn_costs=exp_config.dyn_costs,
        dist_mx=exp_config.dist_mx,
        crossover=exp_config.crossover,
        mutators=exp_config.mutators,
        mut_ps=exp_config.mut_ps,
        crossover_kwargs=exp_config.crossover_kwargs,
        mut_kwargs=exp_config.mut_kwargs,
        initial_vx=0,
        fix_max_retries=1,
        fix_max_add_iters=1000,
        rng=rng,
    )
    generation_n = exp_config.generation_n
    timeout = exp_config.exp_timeout
    early_stop_n = exp_config.early_stop_n
    t0 = time()
    population, initial_data, rng = next(stepper)  # type: ignore
    data = {}
    process_generation_data(initial_data, data)
    min_cost = min(initial_data.costs)
    min_ix = next(ix for ix, cost in enumerate(initial_data.costs) if cost == min_cost)
    best_sol = population[min_ix]
    end_experiment = partial(write_results, data=data, path=_results_path)
    iters_since_update = 0
    for next_gen, next_gen_data, rng in take(generation_n, stepper):
        if time() - t0 > timeout:
            end_experiment(best_sol, ExpEndReason.TIMEOUT)
            return
        if iters_since_update >= early_stop_n:
            end_experiment(best_sol, ExpEndReason.EARLY_STOP_N)
        process_generation_data(next_gen_data, data)
        current_min_obj = min(next_gen_data.costs)
        if current_min_obj < min_cost:
            min_cost = current_min_obj
            iters_since_update = 0
            best_sol_ix = next(
                ix for ix, cost in enumerate(next_gen_data.costs) if cost == min_cost
            )
            best_sol = next_gen[best_sol_ix]
        else:
            iters_since_update += 1

    end_experiment(best_sol, ExpEndReason.ITERS)
