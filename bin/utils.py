from pathlib import Path
from typing import Any, Optional, Union, TypeVar
import json
from datetime import datetime
from traceback import format_tb
from enum import Enum, auto
from math import isfinite
import os

import numpy as np

from libs.optimizers.algorithms.genetic.steppers.utils import NextGenData
from libs.environment.cost_calculators import normalize_obj_max, normalize_obj_min
from libs.data_loading.utils import ExperimentType
from bin.progress_bar import EndReason


Rng = TypeVar("Rng", bound=np.random.Generator)


def write_results(
    best_sol: np.ndarray,
    end_reason: EndReason,
    data: dict[str, Any],
    exp_conf_path: Union[str, Path],
    results_path: Union[str, Path],
    end_iter: int,
    exec_time: float,
    map_path: Union[str, Path],
    exception: Optional[Exception] = None,
):
    results_path = Path(results_path)
    # add process id to the file name before extension
    parts = results_path.parts
    filename_parts = parts[-1].split(".")
    filename_parts[-2] = f"{filename_parts[-2]}_pid-{os.getpid()}"
    filename = ".".join(filename_parts)
    results_path = Path(*parts[:-1], filename)
    data["end_reason"] = end_reason.name.lower()
    data["best_sol"] = best_sol.tolist()
    data["experiment_config_path"] = str(exp_conf_path)
    data["end_iter"] = end_iter
    data["exec_time"] = exec_time
    # environment path
    data["map_path"] = str(map_path)
    if exception is not None:
        sep = ", "
        data[
            "exception"
        ] = f"{type(exception).__name__}: {sep.join(map(str, exception.args))}"
        data["traceback"] = "\n".join(format_tb(exception.__traceback__))
    with results_path.open("w") as f:
        json.dump(data, f)


def process_generation_data(next_gen_data: NextGenData, data: dict[str, Any]):
    """
    Mutated `data` dict.
    """

    costs_arr = np.array(next_gen_data.costs)
    costs_arr: np.ndarray = costs_arr[np.isfinite(costs_arr)]
    if len(costs_arr) > 0:
        mean = costs_arr.mean()
        std = costs_arr.std()
        min_cost = costs_arr.min()
    else:
        NaN = float("NaN")
        mean = NaN
        std = NaN
        min_cost = NaN

    _costs_k = "costs"
    _mean_k = "mean"
    _std_dev_k = "std_dev"
    _current_best_k = "current_best"
    _no_of_fix_failures_k = "no_of_fix_failures"
    _mutation_p_k = "mutation_p"
    _crossover_inv_p_k = "crossover_inv_p"

    if not data:
        data_costs = {}
        data_costs[_mean_k] = [mean]
        data_costs[_std_dev_k] = [std]
        data_costs[_current_best_k] = [min_cost]
        data[_costs_k] = data_costs
        data[_no_of_fix_failures_k] = [next_gen_data.no_of_fix_failures]
        data[_mutation_p_k] = [next_gen_data.mutation_p]
        data[_crossover_inv_p_k] = [next_gen_data.crossover_inv_p]
    else:
        data_costs = data[_costs_k]
        data_costs[_mean_k].append(mean)
        data_costs[_std_dev_k].append(std)
        data_costs[_current_best_k].append(min_cost)
        data[_no_of_fix_failures_k].append(next_gen_data.no_of_fix_failures)
        data[_mutation_p_k].append(next_gen_data.mutation_p)
        data[_crossover_inv_p_k].append(next_gen_data.crossover_inv_p)


def get_rand_exp_map(
    exp_t: ExperimentType, dir_path: Union[str, Path], rng: Rng
) -> tuple[Path, Rng]:
    """
    Draws from files in `dir_path` by `exp_t`.
    """

    file_exp_types = ["tsp"] if exp_t == ExperimentType.TSP else ["vrp", "vrpp", "irp"]
    files = [
        f for f in Path(dir_path).iterdir() if f.is_file() and str(f).endswith(".json")
    ]
    file_pool = [f for f in files if any(et in str(f) for et in file_exp_types)]
    if not file_pool:
        raise NoFilesError(
            f"no files for experiment type {exp_t.name.lower()} at {dir_path}"
        )
    selected: Path = rng.choice(file_pool)
    return selected, rng


def get_datetime_str() -> str:
    return datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


class NoFilesError(Exception):
    ...


class NormMode(Enum):
    MIN = auto()
    MAX = auto()


class ObjectiveNormalizer:
    def __init__(self, mode: NormMode) -> None:
        super().__init__()
        self.highest: float = -float("inf")
        self.lowest: float = float("inf")
        self._normalizer = (
            normalize_obj_min if mode == NormMode.MIN else normalize_obj_max
        )

    def normalize(self, new_obj: float) -> float:
        if isfinite(new_obj):
            if new_obj > self.highest:
                self.highest = new_obj
            elif new_obj < self.lowest:
                self.lowest = new_obj
        return self._normalizer(new_obj, self.highest, self.lowest)
