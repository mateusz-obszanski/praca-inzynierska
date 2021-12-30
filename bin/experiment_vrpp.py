from pathlib import Path
from time import time

import numpy as np
from bin.progress_bar import (
    ExpProgressShow,
    EndReason,
    EndExperiment,
    ExpProgressSilent,
)

from bin.utils import process_generation_data, write_results
from libs.data_loading.loaders import get_experiment_config
from libs.optimizers.algorithms.genetic.steppers.vrpp import genetic_stepper_vrpp
from libs.data_loading.utils import ExperimentType


Chromosome = np.ndarray
Population = list[Chromosome]


def experiment_vrpp(
    exp_conf_path: str,
    results_path: str,
    population_size: int,
    generation_n: int,
    exp_timeout: int,
    early_stop_n: int,
    salesmen_n: int,
    fillval: int,
    silent: bool = True,
):
    """
    ExpProgress is responsible for raising EndExperiment exception.
    """

    i = -1
    t0 = time()
    try:
        _results_path = Path(results_path)
        del results_path
        exp_config = get_experiment_config(
            ExperimentType.VRPP,
            exp_conf_path,
            population_size,
            generation_n,
            exp_timeout,
            early_stop_n,
            salesmen_n,
            fillval,
        )
        rng = np.random.default_rng(exp_config.rng_seed)
        t0 = time()
        stepper = genetic_stepper_vrpp(
            population=exp_config.population,
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
            salesmen_n=salesmen_n,
            demands=exp_config.demands,
            fillval=exp_config.fillval,
        )
        generation_n = exp_config.generation_n
        progress_handler = ExpProgressSilent if silent else ExpProgressShow
        with progress_handler(
            total_iters=generation_n,
            timeout=exp_config.exp_timeout,
            early_stop=exp_config.early_stop_n,
        ) as progress:
            population, initial_data, rng = next(stepper)  # type: ignore
            data = {}
            process_generation_data(initial_data, data)
            min_cost = min(initial_data.costs)
            min_ix = next(
                ix for ix, cost in enumerate(initial_data.costs) if cost == min_cost
            )
            best_sol = population[min_ix]
            i = 0
            # for i, (next_gen, next_gen_data, rng) in enumerate(take(generation_n, stepper)):
            for i, (next_gen, next_gen_data, rng) in enumerate(stepper):
                if i >= generation_n:
                    raise EndExperiment(EndReason.ITERATIONS)
                process_generation_data(next_gen_data, data)
                current_min_obj = min(next_gen_data.costs)
                if current_min_obj < min_cost:
                    min_cost = current_min_obj
                    best_sol_ix = next(
                        ix
                        for ix, cost in enumerate(next_gen_data.costs)
                        if cost == min_cost
                    )
                    best_sol = next_gen[best_sol_ix]
                    progress.reset_early_stop_cnt()
                progress.iteration_update()
    except EndExperiment as end_exp:
        tk = time()
        write_results(
            best_sol,  # type: ignore
            end_exp.reason,
            data,  # type: ignore
            exp_conf_path,
            _results_path,  # type: ignore
            end_iter=i,
            map_path=exp_config.map_path,  # type: ignore
            exec_time=(tk - t0),
        )
        return
    except Exception as e:
        tk = time()
        end_exp_obj = EndExperiment(EndReason.EXCEPTION, exception=e)
        write_results(
            np.array([]),
            end_exp_obj.reason,
            data={},
            exp_conf_path=exp_conf_path,
            results_path=_results_path,  # type: ignore
            end_iter=i,
            map_path=exp_config.map_path,  # type: ignore
            exception=end_exp_obj.exception,
            exec_time=(tk - t0),
        )
        raise e
