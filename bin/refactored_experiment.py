from pathlib import Path

from more_itertools import take
import numpy as np
from bin.progress_bar import ExpProgress, EndReason, EndExperiment

from bin.utils import process_generation_data, write_results
from libs.data_loading.loaders import get_experiment_config
from libs.optimizers.algorithms.genetic.steppers.tsp import genetic_stepper_tsp
from libs.data_loading.utils import ExperimentType


Chromosome = np.ndarray
Population = list[Chromosome]


def experiment_tsp(
    exp_conf_path: str,
    results_path: str,
    population_size: int,
    generation_n: int,
    exp_timeout: int,
    early_stop_n: int,
):
    """
    ExpProgress is responsible for raising EndExperiment exception.
    """

    i = -1
    try:
        _results_path = Path(results_path)
        del results_path
        exp_config = get_experiment_config(
            ExperimentType.TSP,
            exp_conf_path,
            population_size,
            generation_n,
            exp_timeout,
            early_stop_n,
        )
        rng = np.random.default_rng(exp_config.rng_seed)
        stepper = genetic_stepper_tsp(
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
        )
        generation_n = exp_config.generation_n
        with ExpProgress(
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
            print("#################################################")
            # for i, (next_gen, next_gen_data, rng) in enumerate(take(generation_n, stepper)):
            for i, (next_gen, next_gen_data, rng) in enumerate(stepper):
                if i >= generation_n:
                    return
                # FIXME remove
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                print(f"{i = }{40*'-'}")
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
        write_results(
            best_sol, end_exp.reason, data, exp_conf_path, _results_path, end_iter=i  # type: ignore
        )
        return
    except Exception as e:
        end_exp_obj = EndExperiment(EndReason.EXCEPTION, exception=e)
        write_results(
            np.array([]),
            end_exp_obj.reason,
            data={},
            exp_conf_path=exp_conf_path,
            results_path=_results_path,  # type: ignore
            end_iter=i,
            exception=end_exp_obj.exception,
        )
        return
