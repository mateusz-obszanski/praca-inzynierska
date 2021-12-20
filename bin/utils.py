from enum import Enum, auto
from pathlib import Path
from typing import Any, Union
import json

import numpy as np

from libs.optimizers.algorithms.genetic.steppers.utils import NextGenData


class ExpEndReason(Enum):
    TIMEOUT = auto()
    EARLY_STOP_N = auto()
    ITERS = auto()
    """ordinary reason"""


def write_results(
    best_sol: np.ndarray,
    end_reason: ExpEndReason,
    data: dict[str, Any],
    path: Union[str, Path],
):
    path = Path(path)
    data["end_reason"] = end_reason.name.lower()
    data["best_sol"] = best_sol.tolist()
    with path.open("w") as f:
        json.dump(data, f)


def process_generation_data(next_gen_data: NextGenData, data: dict[str, Any]):
    """
    Mutated `data` dict.
    """

    costs_arr = np.array(next_gen_data.costs)
    if not data:
        data_costs = {}
        data_costs["mean"] = [costs_arr.mean()]
        data_costs["std_dev"] = [costs_arr.std()]
        data["costs"] = data_costs
        data["no_of_fix_failures"] = [next_gen_data.no_of_fix_failures]
        data["mutation_p"] = [next_gen_data.mutation_p]
        data["crossover_inv_p"] = [next_gen_data.crossover_inv_p]
    else:
        data["costs"]["mean"].append(costs_arr.mean())
        data["costs"]["std_dev"].append(costs_arr.std())
        data["no_of_fix_failures"].append(next_gen_data.no_of_fix_failures)
        data["mutation_p"].append(next_gen_data.mutation_p)
        data["crossover_inv_p"].append(next_gen_data.crossover_inv_p)
