from dataclasses import dataclass
from typing import Any

import numpy as np

from libs.optimizers.algorithms.genetic.operators.crossovers import CrossoverNDArray
from libs.optimizers.algorithms.genetic.operators.mutations import Mutator


Chromosome = np.ndarray
CostMx = np.ndarray
ExpirTime = float
DistMx = np.ndarray
Rng = np.random.Generator


@dataclass
class ExperimentConfigBase:
    ...


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
    rng: Rng
    generation_n: int
    exp_timeout: int
    early_stop_n: int


@dataclass
class ConfigVRP(ExperimentConfigBase):
    vehicle_n: int


@dataclass
class ConfigVRPP(ExperimentConfigBase):
    ...


@dataclass
class ConfigIRP(ExperimentConfigBase):
    ...
