from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
from bin.exceptions import InitialPopFixError
from libs.data_loading.exp_funcs_map import EXP_ALLOWED_FUNCS

from libs.data_loading.utils import ExperimentType, load_data
from libs.data_loading.base import (
    ConfigTSP,
    ConfigVRP,
    ConfigVRPP,
    ConfigIRP,
    ExperimentConfigBase,
)
from libs.environment.cost_calculators import cost_calc_tsp
from libs.optimizers.algorithms.genetic.operators.fixers import (
    check_chromosome_tsp,
    fix_tsp,
)
from libs.optimizers.algorithms.genetic.operators.mutations import mutate_swap
from libs.solution.initial_solution_creators.random import (
    create_tsp_sol_rand,
    create_vrp_sol_rand,
    create_irp_sol_rand,
)


Rng = np.random.Generator


@overload
def generate_rand_conf(
    exp_t: Literal[ExperimentType.TSP],
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
) -> ConfigTSP:
    ...


@overload
def generate_rand_conf(
    exp_t: Literal[ExperimentType.VRP],
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
) -> ConfigVRP:
    ...


@overload
def generate_rand_conf(
    exp_t: Literal[ExperimentType.VRPP],
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
) -> ConfigVRPP:
    ...


@overload
def generate_rand_conf(
    exp_t: Literal[ExperimentType.IRP],
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
) -> ConfigIRP:
    ...


def generate_rand_conf(
    exp_t: ExperimentType,
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
) -> ExperimentConfigBase:
    # for this function
    rng = np.random.default_rng()
    hyperparams = load_data(Path("bin/hyperparams.json"))
    if exp_t == ExperimentType.TSP:
        return _generate_rand_conf_tsp(
            env_data,
            population_size,
            generation_n,
            timeout,
            early_stop_iters,
            hyperparams,
            rng,
            rng_seed=rng.integers((1 << 32) - 1),
        )
    else:
        raise NotImplementedError(exp_t)


def _generate_rand_conf_tsp(
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
    hyperparams: dict[str, Any],
    rng: Rng,
    rng_seed: int,
) -> ConfigTSP:
    RETRY_N = 100
    initial_vx = env_data.get("initial_vx", 0)
    population = [
        np.array(create_tsp_sol_rand(env_data["dist_mx"], initial_vx, rng)[0])
        for _ in range(population_size)
    ]
    dist_mx = env_data["dist_mx"]
    inv_sol_ixs = [
        ix
        for ix, sol in enumerate(population)
        if not check_chromosome_tsp(
            sol.tolist(), dist_mx, initial_vx, forbidden_val=(-1)
        )
    ]
    if inv_sol_ixs:
        nl = "\n"
        print(f"before fixing: {nl.join(map(str, population))}")
        i = 0
        retries = 0
        while i < len(inv_sol_ixs):
            if retries > RETRY_N:
                # FIXME remove
                print(
                    "\n".join(
                        f"{sol}{'x' if i in inv_sol_ixs else ''}"
                        for i, sol in enumerate(population)
                    )
                )
                inv_transitions = [
                    f"{i}->{j}"
                    for i in range(len(dist_mx))
                    for j in range(len(dist_mx))
                    if dist_mx[i, j] == -1
                ]
                print(f"{inv_transitions = }")
                raise InitialPopFixError
            sol_ix = inv_sol_ixs[i]
            inv_sol = population[sol_ix]
            fixed, fix_status = fix_tsp(
                inv_sol.tolist(),
                dist_mx,
                initial_vx,
                forbidden_val=(-1),
                max_add_iters=1000,
                max_retries=1,
            )
            if fix_status:
                print(f"fixed individual {sol_ix}")
                population[sol_ix] = np.array(fixed, dtype=np.int64)
                retries = 0
            else:
                print(f"will regenerate individual {sol_ix}")
                retries += 1
                new_sol, rng = create_tsp_sol_rand(dist_mx, initial_vx, rng)
                population[sol_ix] = np.array(new_sol, dtype=np.int64)
                continue
            i += 1
        print(f"after fixing: {nl.join(map(str, population))}")
    population = [np.array(c, dtype=np.int64) for c in population]
    f_map = EXP_ALLOWED_FUNCS[ExperimentType.TSP]
    crossover = rng.choice(f_map["crossovers"])
    cross_kws = {
        k: rng.choice(vals)
        for k, vals in hyperparams["crossover_args"][crossover.__name__].items()
    }
    # allowing all mutators
    mutators = f_map["mutators"]
    mut_kws = {mutate_swap: {}}
    mut_ps = {mutate_swap: rng.choice(hyperparams["mutation_rates"])}
    # FIXME remove
    print(f"{cross_kws = }")
    print(f"{mut_kws = }")
    print(f"{mut_ps = }")
    return ConfigTSP(
        population=population,
        dyn_costs=env_data["dyn_costs"],
        dist_mx=env_data["dist_mx"],
        crossover=crossover,
        crossover_kwargs=cross_kws,
        mutators=mutators,
        mut_kwargs=mut_kws,  # type: ignore
        mut_ps=mut_ps,  # type: ignore
        initial_vx=0,
        fix_max_add_iters=1000,
        fix_max_retries=1,
        rng_seed=rng_seed,
        generation_n=generation_n,
        exp_timeout=timeout,
        early_stop_n=early_stop_iters,
    )
