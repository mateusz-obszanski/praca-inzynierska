from typing import Any, Literal, overload

import numpy as np
from libs.data_loading.exp_funcs_map import EXP_ALLOWED_FUNCS

from libs.data_loading.utils import ExperimentType, load_data
from libs.data_loading.base import ConfigTSP, ConfigVRP, ConfigVRPP, ConfigIRP, ExperimentConfigBase
from libs.solution.initial_solution_creators.random import create_tsp_sol_rand, create_vrp_sol_rand, create_irp_sol_rand


@overload
def generate_rand_conf(exp_t: Literal[ExperimentType.TSP]) -> ConfigTSP:
    ...

@overload
def generate_rand_conf(exp_t: Literal[ExperimentType.VRP]) -> ConfigVRP:
    ...

@overload
def generate_rand_conf(exp_t: Literal[ExperimentType.VRPP]) -> ConfigVRPP:
    ...

@overload
def generate_rand_conf(exp_t: Literal[ExperimentType.IRP]) -> ConfigIRP:
    ...

def generate_rand_conf(exp_t: ExperimentType, env_data: dict[str, Any], seed: int = 0, exp_seed: int = 0) -> ExperimentConfigBase:
    """
    `exp_seed` - seed for experiment's random generator
    `seed` - seed for this function's random generator
    """
    rng_exp = np.random.default_rng(seed=exp_seed)
    rng = np.random.default_rng(seed=seed)
    hyperparams = load_data("bin/hyperparams.json")
    if exp_t == ExperimentType.TSP:
        population, rng_exp = create_tsp_sol_rand(env_data["dist_mx"], env_data.get("initial_vx", 0), rng_exp)
        f_map = EXP_ALLOWED_FUNCS[ExperimentType.TSP]
        crossover = rng.choice(f_map["crossovers"])
        cross_kws = {k: rng.choice(vals) for k, vals in hyperparams["crossover_args"][crossover.__name__]}
        # allowing all mutators
        mutators = f_map["mutators"]
        mut_kws
        return ConfigTSP(
            population=population,
            dyn_costs=env_data["dyn_costs"]
            dist_mx=env_data["dist_mx"],
            crossover=crossover,
            crossover_kwargs=cross_kws,
            mutators=mutators,
            mut_kwargs=
        )
    else:
        raise NotImplementedError(exp_t)
