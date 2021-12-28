from pathlib import Path
from typing import Any, Optional, Union
import json
from traceback import format_tb

import numpy as np

from libs.optimizers.algorithms.genetic.steppers.utils import NextGenData
from bin.progress_bar import EndReason


def write_results(
    best_sol: np.ndarray,
    end_reason: EndReason,
    data: dict[str, Any],
    exp_conf_path: Union[str, Path],
    results_path: Union[str, Path],
    end_iter: int,
    exception: Optional[Exception] = None,
):
    results_path = Path(results_path)
    data["end_reason"] = end_reason.name.lower()
    data["best_sol"] = best_sol.tolist()
    data["experiment_config_path"] = str(exp_conf_path)
    data["end_iter"] = end_iter
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
    mean = costs_arr.mean()
    std = costs_arr.std()
    min_cost = costs_arr.min()
    if not data:
        data_costs = {}
        data_costs["mean"] = [mean]
        data_costs["std_dev"] = [std]
        data_costs["best"] = [min_cost]
        data["costs"] = data_costs
        data["no_of_fix_failures"] = [next_gen_data.no_of_fix_failures]
        data["mutation_p"] = [next_gen_data.mutation_p]
        data["crossover_inv_p"] = [next_gen_data.crossover_inv_p]
    else:
        data["costs"]["mean"].append(mean)
        data["costs"]["std_dev"].append(std)
        data["costs"]["best"].append(min_cost)
        data["no_of_fix_failures"].append(next_gen_data.no_of_fix_failures)
        data["mutation_p"].append(next_gen_data.mutation_p)
        data["crossover_inv_p"].append(next_gen_data.crossover_inv_p)
