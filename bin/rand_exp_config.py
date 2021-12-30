from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

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
from libs.optimizers.algorithms.genetic.operators.fixers import (
    check_chromosome_tsp,
    fix_tsp,
    check_transitions,
    fix_vrpp,
)
from libs.optimizers.algorithms.genetic.operators.mutations import (
    mutate_del,
    mutate_insert,
    mutate_swap,
)
from libs.solution.initial_solution_creators.random import (
    create_tsp_sol_rand,
    create_vrp_sol_rand,
    create_irp_sol_rand,
)
from libs.utils.matrix import extend_cost_mx


Rng = np.random.Generator


@overload
def generate_rand_conf(
    exp_t: Literal[ExperimentType.TSP],
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
    map_path: Union[str, Path],
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
    map_path: Union[str, Path],
    salesmen_n: int,
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
    map_path: Union[str, Path],
    salesmen_n: int,
    fillval: int,
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
    map_path: Union[str, Path],
    salesmen_n: int,
    fillval: int,
) -> ConfigIRP:
    ...


def generate_rand_conf(
    exp_t: ExperimentType,
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
    map_path: Union[str, Path],
    salesmen_n: Optional[int] = None,
    fillval: Optional[int] = None,
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
            map_path=map_path,
        )
    elif exp_t == ExperimentType.VRP:
        if salesmen_n is None:
            raise ValueError("`salesmen_n` must be provided")
        return _generate_rand_conf_vrp(
            env_data,
            population_size,
            generation_n,
            timeout,
            early_stop_iters,
            hyperparams,
            rng,
            rng_seed=rng.integers((1 << 32) - 1),
            map_path=map_path,
            salesmen_n=salesmen_n,
        )
    elif exp_t == ExperimentType.VRPP:
        if None in (salesmen_n, fillval):
            raise ValueError("`salesmen_n` and `fillval` must be provided")
        return _generate_rand_conf_vrpp(
            env_data,
            population_size,
            generation_n,
            timeout,
            early_stop_iters,
            hyperparams,
            rng,
            rng_seed=rng.integers((1 << 32) - 1),
            map_path=map_path,
            salesmen_n=salesmen_n,  # type: ignore
            fillval=fillval,  # type: ignore
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
    map_path: Union[str, Path],
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
        i = 0
        retries = 0
        while i < len(inv_sol_ixs):
            if retries > RETRY_N:
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
                population[sol_ix] = np.array(fixed, dtype=np.int64)
                retries = 0
            else:
                retries += 1
                new_sol, rng = create_tsp_sol_rand(dist_mx, initial_vx, rng)
                population[sol_ix] = np.array(new_sol, dtype=np.int64)
                continue
            i += 1
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
        map_path=str(map_path),
    )


def _generate_rand_conf_vrp(
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
    hyperparams: dict[str, Any],
    rng: Rng,
    rng_seed: int,
    map_path: Union[str, Path],
    salesmen_n: int,
) -> ConfigVRP:
    RETRY_N = 100
    initial_vx = env_data.get("initial_vx", 0)
    ext_dist_mx = extend_cost_mx(
        env_data["dist_mx"], copy_n=(salesmen_n - 1), to_copy_ix=initial_vx
    )
    ini_and_dummy_vxs = {*range(salesmen_n)}
    population = [
        np.array(
            create_vrp_sol_rand(ext_dist_mx, initial_vx, rng, ini_and_dummy_vxs)[0]
        )
        for _ in range(population_size)
    ]
    inv_sol_ixs = [
        ix
        for ix, sol in enumerate(population)
        if not check_transitions(sol.tolist(), ext_dist_mx, forbid_val=(-1))
    ]
    if inv_sol_ixs:
        i = 0
        retries = 0
        while i < len(inv_sol_ixs):
            if retries > RETRY_N:
                raise InitialPopFixError
            sol_ix = inv_sol_ixs[i]
            inv_sol = population[sol_ix]
            fixed, fix_status = fix_tsp(
                inv_sol.tolist(),
                ext_dist_mx,
                initial_vx,
                forbidden_val=(-1),
                max_add_iters=1000,
                max_retries=1,
            )
            if fix_status:
                population[sol_ix] = np.array(fixed, dtype=np.int64)
                retries = 0
            else:
                retries += 1
                new_sol, rng = create_vrp_sol_rand(
                    ext_dist_mx, initial_vx, rng, ini_and_dummy_vxs
                )
                population[sol_ix] = np.array(new_sol, dtype=np.int64)
                continue
            i += 1
    population = [np.array(c, dtype=np.int64) for c in population]
    f_map = EXP_ALLOWED_FUNCS[ExperimentType.VRP]
    crossover = rng.choice(f_map["crossovers"])
    cross_kws = {
        k: rng.choice(vals)
        for k, vals in hyperparams["crossover_args"][crossover.__name__].items()
    }
    # allowing all mutators
    mutators = f_map["mutators"]
    mut_kws = {mutate_swap: {}}
    mut_ps = {mutate_swap: rng.choice(hyperparams["mutation_rates"])}
    return ConfigVRP(
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
        map_path=str(map_path),
        salesmen_n=salesmen_n,
    )


def _generate_rand_conf_vrpp(
    env_data: dict[str, Any],
    population_size: int,
    generation_n: int,
    timeout: int,
    early_stop_iters: int,
    hyperparams: dict[str, Any],
    rng: Rng,
    rng_seed: int,
    map_path: Union[str, Path],
    salesmen_n: int,
    fillval: int,
) -> ConfigVRPP:
    RETRY_N = 100
    initial_vx = env_data.get("initial_vx", 0)
    ext_dist_mx = extend_cost_mx(
        env_data["dist_mx"], copy_n=(salesmen_n - 1), to_copy_ix=initial_vx
    )
    ini_and_dummy_vxs = {*range(salesmen_n)}
    population = [
        np.array(
            create_vrp_sol_rand(ext_dist_mx, initial_vx, rng, ini_and_dummy_vxs)[0]
        )
        for _ in range(population_size)
    ]
    inv_sol_ixs = [
        ix
        for ix, sol in enumerate(population)
        if not check_transitions(sol.tolist(), ext_dist_mx, forbid_val=(-1))
    ]
    if inv_sol_ixs:
        i = 0
        retries = 0
        while i < len(inv_sol_ixs):
            if retries > RETRY_N:
                raise InitialPopFixError
            sol_ix = inv_sol_ixs[i]
            inv_sol = population[sol_ix]
            fixed, fix_status = fix_vrpp(
                inv_sol.tolist(),
                ext_dist_mx,
                initial_vx,
                forbidden_val=(-1),
                max_add_iters=1000,
                max_retries=1,
                fillval=fillval,
            )
            if fix_status:
                population[sol_ix] = np.array(fixed, dtype=np.int64)
                retries = 0
            else:
                retries += 1
                new_sol, rng = create_vrp_sol_rand(
                    ext_dist_mx, initial_vx, rng, ini_and_dummy_vxs
                )
                population[sol_ix] = np.array(new_sol, dtype=np.int64)
                continue
            i += 1
    population = [np.array(c, dtype=np.int64) for c in population]
    f_map = EXP_ALLOWED_FUNCS[ExperimentType.VRPP]
    crossover = rng.choice(f_map["crossovers"])
    cross_kws = {
        k: rng.choice(vals)
        for k, vals in hyperparams["crossover_args"][crossover.__name__].items()
    }
    # allowing all mutators
    mutators = f_map["mutators"]
    ini_and_dummy_vxs = {*range(salesmen_n)}
    dist_mx = env_data["dist_mx"]
    mut_kws = {
        mutate_swap: {},
        mutate_insert: {
            "rand_range": (min(ini_and_dummy_vxs) + 1, len(dist_mx)),
            "ini_and_dummy_vxs": ini_and_dummy_vxs,
            "fillval": fillval,
        },
        mutate_del: {"fillval": fillval},
    }
    mutation_rates = hyperparams["mutation_rates"]
    mut_ps = {
        m: rng.choice(mutation_rates) for m in (mutate_swap, mutate_insert, mutate_del)
    }
    return ConfigVRPP(
        population=population,
        dyn_costs=env_data["dyn_costs"],
        dist_mx=dist_mx,
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
        map_path=str(map_path),
        salesmen_n=salesmen_n,
        demands=tuple(env_data["demands_vrpp"]),
        fillval=fillval,
    )
