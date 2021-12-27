from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union, overload
import yaml

import numpy as np

from .utils import get_f_by_name, load_data, ExperimentType
from .base import ConfigIRP, ConfigTSP, ConfigVRP, ConfigVRPP, ExperimentConfigBase
from .exp_funcs_map import EXP_ALLOWED_FUNCS
from libs.solution.initial_solution_creators.random import create_tsp_sol_rand, create_vrp_sol_rand, create_irp_sol_rand


class ExperimentParser(ABC):
    @abstractmethod
    def __call__(self, data: dict[str, Any]) -> ExperimentConfigBase:
        ...


class ParserTSP(ExperimentParser):
    def __call__(self, data: dict[str, Any], generation_n: int, exp_timeout: int, early_stop_n: int) -> ConfigTSP:
        super().__call__(data)
        INITIAL_VX = 0
        rng: np.random.Generator = np.random.default_rng(seed=data.get("rng_seed", 0))
        pop_path = data.get("population_path")
        if pop_path is None:
            population = [create_tsp_sol_rand(
                dist_mx=data["dist_mx"],
                initial_vx=INITIAL_VX,
                rng=rng
            )[0] for _ in range(data["population_size"])]
        else:
            population = load_data(pop_path)["population_path"]
        population = [np.array(x, dtype=np.int64) for x in population]
        func_map = EXP_ALLOWED_FUNCS[ExperimentType.TSP]
        return ConfigTSP(
            population=population,
            dyn_costs=data["dyn_costs"],
            dist_mx=data["dist_mx"],
            crossover=get_f_by_name(data["crossover"], func_map["crossovers"]),
            crossover_kwargs=data["crossover_kwargs"],
            mutators=[get_f_by_name(m, func_map["mutators"]) for m in data["mutators"]],
            mut_kwargs=data["mutator_kwargs"],
            mut_ps=data["mut_ps"],
            initial_vx=INITIAL_VX,
            fix_max_add_iters=data.get("fix_max_add_iters", 100),
            fix_max_retries=data.get("fix_max_retries", 1),
            rng=rng,
            generation_n=generation_n,
            exp_timeout=exp_timeout,
            early_stop_n=early_stop_n
        )


class ParserVRP(ExperimentParser):
    def __call__(self, data: dict[str, Any]) -> ConfigVRP:
        return super().__call__(data)


class ParserVRPP(ExperimentParser):
    def __call__(self, data: dict[str, Any]) -> ConfigVRPP:
        return super().__call__(data)


class ParserIRP(ExperimentParser):
    def __call__(self, data: dict[str, Any]) -> ConfigIRP:
        return super().__call__(data)


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.TSP], path: Union[str, Path]
) -> ConfigTSP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.VRP], path: Union[str, Path]
) -> ConfigVRPP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.VRPP], path: Union[str, Path]
) -> ConfigVRPP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.IRP], path: Union[str, Path]
) -> ConfigIRP:
    ...


def get_experiment_config(
    exp_t: ExperimentType, path: Union[str, Path]
) -> ExperimentConfigBase:
    parser_map: dict[ExperimentType, type[ExperimentParser]] = {
        ExperimentType.TSP: ParserTSP,
        ExperimentType.VRP: ParserVRP,
        ExperimentType.VRPP: ParserVRPP,
        ExperimentType.IRP: ParserIRP,
    }
    parser = parser_map[exp_t]()
    with Path(path).open("r") as f:
        exp_data_to_validate = yaml.full_load(f)
    exp_config_data: ExperimentConfigBase = parser(exp_data_to_validate)
    return exp_config_data


def get_env_data(path: Union[str ,Path]) -> dict[str, Any]:
    data: dict[str, Any] = {"environment_path": path}
    env_data = load_data(path)
    gen_setup = env_data["generation_setup"]
    data["dist_mx"] = np.array(gen_setup["dist_mx"], dtype=np.float64)
    data["dyn_costs"] = [(np.array(dc, dtype=np.float64), t) for dc, t in gen_setup["dyn_costs"]]
    data["initial_vx"] = 0
    return data

# def configure_experiment_data(exp_conf: ExperimentConfigTSP) -> dict[str, Any]:
#     data = load_from_dataclass_paths(exp_conf)
#     data.update(
#         (name, val) for name, val, _ in iterate_dataclass(exp_conf) if name not in data
#     )
#     return data
