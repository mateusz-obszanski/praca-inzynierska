from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator
from collections.abc import Sequence
import itertools as it
import more_itertools as mit
import math

import numpy as np

from .....optimizers.algorithms.genetic.population.chromosomes import ChromosomeTSP
from .....utils.iteration import find_doubled_indices


class FixStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NO_FIX_NEEDED = auto()


@dataclass
class FixResult:
    fix_status: FixStatus
    no_of_errors: int = 0

    def __post_init__(self) -> None:
        if self.fix_status in (FixStatus.SUCCESS, FixStatus.NO_FIX_NEEDED):
            if self.no_of_errors:
                raise ValueError(
                    (
                        "cannot set `fix_status` to `FixStatus.SUCCESS` nor"
                        "`FixStatus.NO_FIX_NEEDED` and give nonzero `no_of_errors`"
                    )
                )

            self.no_of_errors = 0


def fix_tsp(
    chromosome: ChromosomeTSP,
    cost_mx: np.ndarray,
    initial_ix: int = 0,
    inplace: bool = False,
) -> tuple[ChromosomeTSP, FixResult]:
    if inplace:
        chromosome = chromosome[:]

    doubled_nodes = find_doubled_indices(chromosome)
    doubles = set(doubled_nodes.keys())
    transitions_possible = _check_transitions(chromosome, cost_mx)

    if not doubles and all(transitions_possible):
        return chromosome, FixResult(FixStatus.SUCCESS)

    all_vxs = set(range(cost_mx.shape[0]))
    leftout_vxs = all_vxs - doubles
    all_double_ixs = it.chain.from_iterable(doubled_nodes.values())
    visited_nodes = list(it.chain([initial_ix], chromosome))

    # replace with the best replacements
    for doubled_ix in all_double_ixs:
        prev_vx = visited_nodes[doubled_ix - 1]
        try:
            # If two vertices have the same cost, the smaller one
            # (they are ints) is chosen.
            _, best_replacement = min(
                (cost, vx)
                for vx in leftout_vxs
                if (cost := cost_mx[prev_vx, vx]) > 0 and math.isfinite(cost)
            )

        except ValueError:
            # replacement is impossible (in place of current double)
            # it will be attempted later in place of the original genetry:
            continue

        previously_double = chromosome[doubled_ix]
        chromosome[doubled_ix] = best_replacement
        leftout_vxs.remove(best_replacement)
        doubles.remove(previously_double)

    if not leftout_vxs:
        no_of_errors = sum(
            not valid_transition
            for valid_transition in _check_transitions(chromosome, cost_mx)
        )

        if no_of_errors == 0:
            return chromosome, FixResult(FixStatus.SUCCESS)

        return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

    # there were impossible replacements - attempt to replace in place
    # of the original genes

    all_vxs = list(all_vxs)

    for left_double in doubles:
        repl_ix = next(ix for ix, vx in enumerate(all_vxs) if vx == left_double)
        if repl_ix == 0:
            # the first element is the starting point - can not change that
            return chromosome, FixResult(FixStatus.FAILURE)

        prev_vx = repl_ix - 1

        try:
            _, best_replacement = min(
                (cost, vx)
                for vx in leftout_vxs
                if (cost := cost_mx[prev_vx, vx]) > 0 and math.isfinite(cost)
            )
        except ValueError:
            # min got empty iterator - no possible replacements
            # doubles mapping has not been mutated so it is used
            no_of_errors = len(doubled_nodes) + sum(
                not valid_transition
                for valid_transition in _check_transitions(chromosome, cost_mx)
            )
            return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

        chromosome[repl_ix] = best_replacement
        leftout_vxs.remove(best_replacement)

    return chromosome, FixResult(FixStatus.SUCCESS)


def _check_transitions(seq: Sequence[int], cost_mx: np.ndarray, initial_ix: int = 0) -> Generator[bool, None, None]:
    return (
        cost > 0 and math.isfinite(cost)
        for cost in (cost_mx[i, j] for i, j in mit.windowed(it.chain([initial_ix], seq), n=2))
    )
