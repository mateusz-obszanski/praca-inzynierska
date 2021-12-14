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
from typing import Any, Callable, Generator, Optional, TypeVar, TypedDict, Union
from collections.abc import Iterable, Sequence
from pathlib import Path
import yaml
import pandas as pd
import json
import pickle
import itertools as it

from marshmallow import Schema, ValidationError
import numpy as np

from libs.optimizers.algorithms.genetic.population.natural_selection import (
    NaturalSelector,
    replace_invalid_offspring,
)
from libs.schemas.exp_funcs_map import ExperimentType
from libs.schemas.experiment_base import ExperimentBase


from ..libs.solution import SolutionTSP
from ..libs.environment.cost_calculators import (
    CostCalculator,
    CostGenCreator,
    CostT,
    DistMx,
    DynCosts,
    cost_gen_tsp,
)
from ..libs.optimizers.algorithms.genetic.population.parent_selectors import (
    ParentSelector,
    ParentSelectorDeprecated,
    select_best_parents_with_probability,
)
from ..libs.optimizers.algorithms.genetic.operators.fixers import Fixer
from ..libs.optimizers.algorithms.genetic.operators.mutations import Mutator
from ..libs.optimizers.algorithms.genetic.operators.crossovers import (
    Crossover,
    CrossoverNDArray,
)
from ..libs.schemas import ExperimentTSPSchema
from ..libs.schemas.experiment_tsp import ExperimentConfigTSP


@dataclass(frozen=True)
class OffspringGenerationData:
    costs: list[CostT]
    no_of_fix_failures: int
    mutation_p: dict[Mutator, float]
    crossover_inv_p: Optional[float] = None


Chromosome = Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
Population = list[Chromosome]
BestSolution = SolutionTSP
BestCost = CostT
Rng = np.random.Generator
ExpStepper = Generator[tuple[Population, OffspringGenerationData, Rng], None, None]
CostMx = np.ndarray
InitialVx = int
ForbiddenVal = float


def apply_mutators(
    c: np.ndarray,
    mutators: Iterable[Mutator],
    mut_ps: dict[Mutator, float],
    mut_kwargs: Iterable[dict[str, Any]],
    rng: Rng,
) -> tuple[np.ndarray, Rng]:
    for m, kwds in zip(mutators, mut_kwargs):
        c, rng = m(c, mut_ps[m], rng, **kwds)
    return c, rng


def genetic_stepper(
    population: Population,
    dyn_costs: DynCosts,
    dist_mx: DistMx,
    parent_selector: ParentSelector,
    crossover: CrossoverNDArray,
    mutators: list[Mutator],
    fixer: Fixer,
    natural_selector: Optional[NaturalSelector],
    cost_calc: CostCalculator,
    checker: Callable[[np.ndarray, DistMx, InitialVx, ForbiddenVal], bool],
    mut_ps: dict[Mutator, float],
    crossover_kwargs: dict[str, Any],
    mut_kwargs: list[dict[str, Any]],
    cost_calc_kwargs: dict[str, Any],
    fixer_kwargs: dict[str, Any],
    initial_vx: int,
    salesman_v: float,
    fix_max_add_iters: int,
    fix_max_retries: int,
    rng: Rng,
) -> ExpStepper:
    """
    For the first time yields initial population, its data and rng

    If `natural_selector` is `None`, then offspring becomes the next generation.
    """

    # [(mx, exp_t)] -> (mx, exp_t) -> mx -> mx[0, 0]
    forbidden_val = dyn_costs[0][0][0, 0]
    if isinstance(population, tuple):
        _population: list[tuple[np.ndarray, np.ndarray]] = population[0]  # type: ignore

    else:
        _population: list[np.ndarray] = population  # type: ignore
    fix_statuses = [checker(c, dist_mx, initial_vx, forbidden_val) for c in _population]
    if any(not valid for valid in fix_statuses):
        inv_ixs = [i for i, success in enumerate(fix_statuses) if not success]
        raise InitialPopInvalidError("some initial individuals vere not valid", inv_ixs)
    costs = [
        cost_calc(
            [vx for vx in c],
            dyn_costs,
            dist_mx,
            salesman_v,
            forbidden_val=forbidden_val,
            **cost_calc_kwargs,
        )
        for c in _population
    ]
    cross_inv_p = crossover_kwargs.get("inversion_p")
    no_of_failures = sum(not success for success in fix_statuses)
    initial_data = OffspringGenerationData(costs, no_of_failures, mut_ps, cross_inv_p)
    yield _population, initial_data, rng  # type: ignore
    while True:
        parents, parent_costs, rng = parent_selector(_population, costs, rng)
        offspring = list(
            it.chain.from_iterable(
                crossover(p1, p2, rng, **crossover_kwargs)[:-1] for p1, p2 in parents
            )
        )
        mutated_offspring = [
            apply_mutators(c, mutators, mut_ps, mut_kwargs, rng)[0] for c in offspring
        ]
        fixed_results = [
            fixer(c, dist_mx, rng, forbidden_val, fix_max_add_iters, fix_max_retries, **fixer_kwargs)
            for c in mutated_offspring
        ]
        fixed_offspring = [r[0] for r in fixed_results]
        fix_statuses = [r[1] for r in fixed_results]
        checked_offspring = replace_invalid_offspring(
            parents, fixed_offspring, parent_costs, fix_statuses
        )
        if natural_selector is not None:
            offspring_costs = [
                cost_calc(
                    [vx for vx in c],
                    dyn_costs,
                    dist_mx,
                    salesman_v,
                    forbidden_val=forbidden_val,
                )
                for c in checked_offspring
            ]
            flattened_parents = list(it.chain.from_iterable(parents))
            next_gen, next_gen_costs, rng = natural_selector(
                flattened_parents, checked_offspring, parent_costs, offspring_costs, rng
            )
        else:
            next_gen = checked_offspring
            next_gen_costs = [
                cost_calc(
                    [vx for vx in c],
                    dyn_costs,
                    dist_mx,
                    salesman_v,
                    forbidden_val=forbidden_val,
                )
                for c in next_gen
            ]
        no_of_failures = sum(not success for success in fix_statuses)
        generation_data = OffspringGenerationData(
            next_gen_costs, no_of_failures, mut_ps, cross_inv_p
        )
        yield next_gen, generation_data, rng  # type: ignore

        parents = next_gen
        costs = next_gen_costs


class NoExtensionError(Exception):
    """
    Raised when no extension in path present but is required.
    """


class ExpError(Exception):
    ...


class InitialPopInvalidError(ExpError):
    ...


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


def get_experiment_tsp_config(exp_t: ExperimentType, path: Union[str, Path]) -> ExperimentBase:
    schema_map = {
        ExperimentType.TSP: ExperimentTSPSchema,
    }
    exp_schema = schema_map[exp_t]()
    with Path(path).open("r") as f:
        exp_data_to_validate = yaml.full_load(f)
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


def experiment(exp_t: str, exp_conf_path: str, results_path: str):
    # TODO read fields from config file
    # TODO save config file and raw data to results
    # TODO export as CLI
    _results_path = Path(results_path)
    del results_path
    _exp_t = ExperimentType[exp_t]
    del exp_t
    exp_config = get_experiment_tsp_config(_exp_t, exp_conf_path)
    population: Population = [individual["vx_seq"] for individual in exp_config.initial_population]
    stepper = genetic_stepper(
        _population=population,
        dyn_costs=exp_config.dyn_costs
    )
