from dataclasses import dataclass
from typing import Any, Union
from enum import Enum
import json
from pathlib import Path

import numpy as np

from libs.optimizers.algorithms.genetic.operators.crossovers import CrossoverNDArray
from libs.optimizers.algorithms.genetic.operators.mutations import Mutator
from libs.utils.iteration import iterate_dataclass


Chromosome = np.ndarray
CostMx = np.ndarray
ExpirTime = float
DistMx = np.ndarray
Rng = np.random.Generator


@dataclass
class ExperimentConfigBase:
    def data_for_json(self) -> dict[str, Any]:
        d = {name: val for name, val, _ in iterate_dataclass(self)}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, Enum):
                d[k] = v.name
            elif k == "population":
                d[k] = [a.tolist() for a in v]
            elif k == "dyn_costs":
                d[k] = [(mx.tolist(), t) for mx, t in v]
            elif k == "crossover":
                d[k] = v.__name__
            elif k == "crossover_kwargs":
                d[k] = {ck: self.convert_numpy(cv) for ck, cv in v.items()}
            elif k == "mutators":
                d[k] = [m.__name__ for m in v]
            elif k == "mut_kwargs":
                d[k] = {
                    mk.__name__: {ak: self.convert_numpy(av) for ak, av in mv.items()}
                    for mk, mv in v.items()
                }
            elif k == "mut_ps":
                d[k] = {mk.__name__: mv for mk, mv in v.items()}
            elif k == "rng_seed":
                d[k] = int(v)
        return d

    def save_to_json(self, path: Union[str, Path]):
        data = self.data_for_json()
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def convert_numpy(v):
        t_name = type(v).__name__
        if "int" in t_name:
            return int(v)
        elif "float" in t_name:
            return float(v)
        else:
            return v


@dataclass
class ConfigTSP(ExperimentConfigBase):
    population: list[Chromosome]
    dyn_costs: list[tuple[CostMx, ExpirTime]]
    dist_mx: DistMx
    crossover: CrossoverNDArray
    crossover_kwargs: dict[str, Any]
    mutators: list[Mutator]
    mut_kwargs: dict[Mutator, dict[str, Any]]
    mut_ps: dict[Mutator, float]
    initial_vx: int
    fix_max_add_iters: int
    fix_max_retries: int
    rng_seed: int
    generation_n: int
    exp_timeout: int
    early_stop_n: int

    def data_to_json(self) -> dict[str, Any]:
        return super().data_for_json()


@dataclass
class ConfigVRP(ExperimentConfigBase):
    vehicle_n: int


@dataclass
class ConfigVRPP(ExperimentConfigBase):
    ...


@dataclass
class ConfigIRP(ExperimentConfigBase):
    ...
