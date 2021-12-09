# TODO fixers should return FixStatus only
# TODO population generation, tournament selection
# TODO TSP experiment run
# TODO mutations: insertion, deletion (associated with below vvv)
# TODO chromosome sparsing with -1s for crossover (to longer)
# TODO costs with time cutoff
# TODO costs for MVMTSP - assign demand (package num) for cargo at each vertex and reward for its fulfillment (1 visit - 1 package)
# TODO costs for IRP
# TODO simulated annealing
# TODO simulated annealing - change mutators and crossovers appropriately (additional dimension for cargo quantity in chromosomes)
# TODO IRP run

from dataclasses import dataclass, fields
from typing import Any, Generator, Optional, TypeVar, TypedDict, Union
from pathlib import Path
import yaml
import pandas as pd
import json
import pickle

from marshmallow import Schema, ValidationError
import numpy as np


from ..libs.solution import SolutionTSP
from ..libs.environment.cost_calculators import CostGenCreator, CostT, cost_gen_tsp
from ..libs.optimizers.algorithms.genetic.population import Population, PopulationTSP
from ..libs.optimizers.algorithms.genetic.population.generators import (
    PopulationGenerationData,
    generate_population,
)
from ..libs.optimizers.algorithms.genetic.population.population_selectors import (
    PopulationSelector,
    select_population_with_probability,
)
from ..libs.optimizers.algorithms.genetic.population.parent_selectors import (
    ParentSelectorDeprecated,
    select_best_parents_with_probability,
)
from ..libs.optimizers.algorithms.genetic.operators.fixers import Fixer
from ..libs.optimizers.algorithms.genetic.operators.mutations import Mutator
from ..libs.optimizers.algorithms.genetic.operators.crossovers import Crossover
from ..libs.schemas import ExperimentTSPSchema
from ..libs.schemas.experiment_tsp import ExperimentConfigTSP


BestSolution = SolutionTSP
BestCost = CostT
Rng = TypeVar("Rng", bound=np.random.Generator)
StepperTSP = Generator[tuple[Population, PopulationGenerationData, Rng], None, None]


def genetic_tsp_gen(
    population: PopulationTSP,
    cost_mx: np.ndarray,
    parent_selector: ParentSelectorDeprecated,
    population_selector: PopulationSelector,
    crossover: Crossover,
    chromosome_fixer: Fixer,
    mutators: list[Mutator],
    invalidity_weight: float,
    error_weight: float,
    cost_weight: float,
    cost_gen: CostGenCreator,
    rng: np.random.Generator,
) -> StepperTSP:
    while True:
        parents = parent_selector(population, costs)

        yield new_population, generation_data, rng  # type: ignore


class NoExtensionError:
    """
    Raised when no extension in path present but is required.
    """


_FILE_READER = {
    "yaml": yaml.full_load,
    "json": json.load,
    "pkl": pickle.load,
    "pickle": pickle.load,
    "csv": pd.read_csv,
}

_READ_IN_BYTE_MODE = {"pickle": True, "pkl": True}


def load_data(path: Union[str, Path]) -> Any:
    path = Path(path)
    extension = path.suffix[1:]
    reader = _FILE_READER[extension]
    read_mode = f"r{'b' if _READ_IN_BYTE_MODE.get(extension, False) else ''}"
    with path.open(read_mode) as f:
        return reader(f)


def get_experiment_tsp_config(path: Union[str, Path]) -> ExperimentConfigTSP:
    with Path(path).open("r") as f:
        exp_data_to_validate = yaml.full_load(f)
    exp_schema = ExperimentTSPSchema()
    exp_config_data: ExperimentConfigTSP = exp_schema.load(exp_data_to_validate)
    return exp_config_data


FieldName = str
FieldVal = Any
FieldType = type


def iterate_dataclass(
    instance,
) -> Generator[tuple[FieldName, FieldVal, FieldType], None, None]:
    for field in fields(instance):
        name = field.name
        value = getattr(instance, name)
        yield name, value, type(value)


def load_from_dataclass_paths(instance) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, path, ftype in iterate_dataclass(instance):
        if not issubclass(ftype, (Path, str)):
            continue
        loaded[name] = load_data(path)
    return loaded


def configure_experiment_data(exp_conf: ExperimentConfigTSP) -> dict[str, Any]:
    data = load_from_dataclass_paths(exp_conf)
    data.update(
        (name, val) for name, val, _ in iterate_dataclass(exp_conf) if name not in data
    )
    return data


def experiment_tsp(exp_conf_path: str, results_path: str):
    # TODO read fields from config file
    # TODO save config file and raw data to results
    # TODO export as CLI
    exp_config = get_experiment_tsp_config(exp_conf_path)
