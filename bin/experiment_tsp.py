from collections import deque
from pathlib import Path
from time import time
from copy import deepcopy
from typing import Any, TypeVar
from math import isfinite

import numpy as np
from bin.progress_bar import (
    ExpProgressShow,
    EndReason,
    EndExperiment,
    ExpProgressSilent,
)

from bin.utils import process_generation_data, write_results
from libs.data_loading.loaders import (
    get_experiment_config,
    get_experiment_config_enhanced,
)
from libs.optimizers.algorithms.genetic.steppers.tsp import (
    genetic_stepper_tsp,
    genetic_stepper_tsp_enhanced,
)
from libs.data_loading.utils import ExperimentType
from libs.utils.random import shuffle_ensure_change


Chromosome = np.ndarray
Population = list[Chromosome]
Rng = TypeVar("Rng", bound=np.random.Generator)


def experiment_tsp(
    exp_conf_path: str,
    results_path: str,
    population_size: int,
    generation_n: int,
    exp_timeout: int,
    early_stop_n: int,
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
            ExperimentType.TSP,
            exp_conf_path,
            population_size,
            generation_n,
            exp_timeout,
            early_stop_n,
        )
        rng = np.random.default_rng(exp_config.rng_seed)
        t0 = time()
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


def experiment_tsp_enhanced(
    exp_conf_path: str,
    results_path: str,
    population_size: int,
    generation_n: int,
    exp_timeout: int,
    early_stop_n: int,
    adaptation_period: int,
    adaptation_step: float,
    migration_ratio: float,
    migration_period: int,
    partition_n: int,
    silent: bool = True,
):
    """
    ExpProgress is responsible for raising EndExperiment exception.
    """

    assert 0 <= migration_ratio < 1
    assert migration_period > 0
    assert partition_n > 1

    t0 = time()
    _results_path = Path(results_path)
    del results_path
    exp_config = get_experiment_config_enhanced(
        ExperimentType.TSP,
        exp_conf_path,
        population_size,
        generation_n,
        exp_timeout,
        early_stop_n,
        adaptation_period,
        adaptation_step,
        migration_ratio,
        migration_period,
        partition_n,
    )
    rng = np.random.default_rng(exp_config.rng_seed)
    pop_len = len(exp_config.population)
    to_migrate = max(1, min(pop_len, int(round(migration_ratio * pop_len))))
    # partitions - roups that evolve separately
    partition_populations: deque[dict] = deque(
        {"pop": deepcopy(exp_config.population), "costs": None}
        for _ in range(partition_n)
    )
    total_i = 0
    processed_partitions = 0
    t0 = time()
    while True:
        try:
            population = partition_populations[0]["pop"]
            stepper = genetic_stepper_tsp_enhanced(
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
                adaptation_period=exp_config.adaptation_period,
                adaptation_step=exp_config.adaptation_step,
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
                for i, (next_gen, next_gen_data, rng) in enumerate(stepper):
                    if total_i >= generation_n:
                        raise EndExperiment(EndReason.ITERATIONS)
                    if i >= migration_period:
                        raise MigrationSig
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
                    total_i += 1
        except MigrationSig:
            processed_partitions += 1
            partition_populations[0] = {"pop": next_gen, "costs": next_gen_data.costs}  # type: ignore
            partition_populations.rotate()
            if processed_partitions < partition_n:
                continue
            processed_partitions = 0
            partition_populations = migrate_between(partition_populations, to_migrate)

        except EndExperiment as end_exp:
            tk = time()
            write_results(
                best_sol,  # type: ignore
                end_exp.reason,
                data,  # type: ignore
                exp_conf_path,
                _results_path,  # type: ignore
                end_iter=total_i,  # type: ignore
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
                end_iter=total_i,  # type: ignore
                map_path=exp_config.map_path,  # type: ignore
                exception=end_exp_obj.exception,
                exec_time=(tk - t0),
            )
            raise e


class MigrationSig(Exception):
    ...


def migrate_between(
    partitions: deque[dict[str, Any]], to_migrate_n: int
) -> deque[dict[str, Any]]:
    to_migrate_ixs = []
    to_migrate = []
    for p in partitions:
        pop = p["pop"]
        costs = p["costs"]
        finite_cost_ixs = [i for i, c in enumerate(costs) if isfinite(c)]
        sorted_finite_ixs = sorted(finite_cost_ixs, key=lambda i: costs[i])
        migration_candidate_ixs = sorted_finite_ixs[:to_migrate_n]
        to_fill = to_migrate_n - len(migration_candidate_ixs)
        if to_fill > 0:
            # fill with invalid cost individuals
            migration_candidate_ixs.extend(
                pop[i] for i in range(len(pop)) if i not in finite_cost_ixs
            )
        migration_candidates = [pop[i] for i in migration_candidate_ixs]
        to_migrate_ixs.append(migration_candidate_ixs)
        to_migrate.append(migration_candidates)
    # circular rotation migration
    # take from the next one
    for i in range(len(partitions) - 1):
        ixs = to_migrate_ixs[i]
        migrants = to_migrate[i]
        for mi, m in zip(ixs, migrants):
            partitions[i][mi] = m
    for mi, m in zip(to_migrate_ixs[-1], to_migrate[-1]):
        partitions[-1][mi] = m
    return partitions
