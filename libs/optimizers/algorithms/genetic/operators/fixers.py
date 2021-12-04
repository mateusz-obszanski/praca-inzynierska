from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, Optional, Protocol, Generic, TypeVar
from collections.abc import Sequence
import itertools as it
import more_itertools as mit
import math
import sys

import numpy as np

from libs.environment.utils import check_transition, find_invalid_transitions

from .....optimizers.algorithms.genetic.population.chromosomes import ChromosomeTSP
from .....utils.iteration import (
    find_all_occurence_indices,
    find_doubled_indices,
)


Rng = TypeVar("Rng", bound=np.random.Generator)


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


class Fixer(Protocol):
    Chromosome = TypeVar("Chromosome")

    def __call__(
        self,
        chromosome: Chromosome,
        cost_mx: np.ndarray,
        rng: Rng,
        forbidden_val: float = -1,
        max_additional_iterations: int = sys.maxsize,
        *args,
        **kwargs,
    ) -> tuple[Chromosome, FixResult, Rng]:
        ...


def fix_chromosome_no_doubles(
    chromosome: list[int],
    cost_mx: np.ndarray,
    initial_vx: int,
    forbidden_val: float = -1,
    max_add_iters: int = sys.maxsize,
    inplace: bool = False,
) -> tuple[list[int], FixResult]:
    """
    Assumes that the first entry in chromosome is the initial vertex and that
    chromosome ends with it. If not, fails.

    :param max_add_iters int: additional iterations for fixing
    """

    new_chromosome = chromosome if inplace else [*chromosome]
    del chromosome
    if initial_vx != new_chromosome[0] != new_chromosome[-1]:
        print("chromosome should begin and end at `initial_vx`")
        return new_chromosome, FixResult(FixStatus.FAILURE,no_of_errors=(-1))
    occurences_map = find_all_occurence_indices(new_chromosome)
    print(f"{occurences_map = }")
    mult_occ_map = {vx: occ for vx, occ in occurences_map.items() if len(occ) > 1 and vx != initial_vx}
    print(f"{mult_occ_map = }")
    forbidden_transitions = find_invalid_transitions(
        new_chromosome, cost_mx, forbidden_val
    )
    print(f"{forbidden_transitions = }")
    inv_dests = [ft[0][1] for ft in forbidden_transitions]
    print(f"{inv_dests = }")
    inv_dest_ixs = [ft[1][1] for ft in forbidden_transitions]
    print(f"{inv_dest_ixs = }")
    mult_vxs = [*mult_occ_map.keys()]
    print(f"{mult_vxs = }")
    # mistakes - invalid destinations and multiple-occuring vertices
    mistake_ixs = [
        *set(it.chain(inv_dest_ixs, it.chain.from_iterable(mult_occ_map.values())))
    ]
    print(f"{mistake_ixs = }")
    alts = [vx for vx in it.chain(mult_vxs, inv_dests)]
    print(f"{alts = }")
    # alternatives for each invalid destinations and multiple-occuring vertex
    alts_at = {ix: [*alts] for ix in mistake_ixs}
    print(f"{alts_at = }")
    already_used_alts: list[Optional[int]] = [None for _ in mistake_ixs]
    print(f"{already_used_alts = }")
    next_alt_ix_at = [0 for _ in mistake_ixs]
    print(f"{next_alt_ix_at = }")
    mistake_ix = 0
    print(f"{mistake_ix = }")
    iter_n = 0
    print(f"{iter_n = }")
    mistake_n = len(mistake_ixs)
    print(f"{mistake_n = }")
    max_iter_n = mistake_n + max_add_iters
    print(f"{max_iter_n = }")
    while mistake_ix < mistake_n:
        print(f"***********{mistake_ix = }")
        if iter_n == max_iter_n:
            print("failure: max iter_n")
            return new_chromosome, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
        iter_n += 1
        unused_alts = alts_at[mistake_ix][next_alt_ix_at[mistake_ix] :]
        print(f"{unused_alts = }")
        prev_vx_ix = mistake_ixs[mistake_ix] - 1
        print(f"{prev_vx_ix = }")
        prev_vx = new_chromosome[prev_vx_ix]
        print(f"{prev_vx = }")
        mistake_alts = sorted(unused_alts, key=lambda dest: cost_mx[prev_vx, dest])
        print(f"{mistake_alts = }")
        # select next valid alt
        valid_alt = next(
            (
                alt
                for alt in mistake_alts
                if check_transition(prev_vx, alt, cost_mx, forbidden_val)
                and alt not in already_used_alts  # avoids vertex repetition
            ),
            None,
        )
        if valid_alt is None:
            print("no valid alt")
            # fallback to the previous mistake index
            if mistake_ix == 0:
                print("failure: cannot fallback from the beginning")
                # fallback impossible => fixing impossible
                return new_chromosome, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
            print("fallback")
            already_used_alts[mistake_ix] = None
            mistake_ix -= 1
            continue
        print(f"{valid_alt = }")
        already_used_alts[mistake_ix] = valid_alt
        next_alt_ix_at[mistake_ix] += 1
        new_chromosome[mistake_ix] = valid_alt
        print(f"current {new_chromosome = }")
    return new_chromosome, FixResult(FixStatus.SUCCESS)


def fix_chromosome_no_doubles_deprecated(
    chromosome: list[int],
    cost_mx: np.ndarray,
    rng: Rng,
    forbidden_val: float = -1,
    max_additional_iterations: int = 10,
    inplace: bool = False,
) -> tuple[list[int], Rng, FixResult]:
    occurences_map = find_all_occurence_indices(chromosome)
    is_doubled = {v: len(occurences) > 1 for v, occurences in occurences_map.items()}
    # list of (a, b), (ix of a in chromosome, ix of b in chromosome) if forbidden
    forbidden_transitions = find_invalid_transitions(chromosome, cost_mx, forbidden_val)
    # create alternatives for substitution of b in invalid transitions a -> b
    # sorted cost rows filtered by transition validity
    vertices = list(range(cost_mx.shape[0]))
    alternatives = {
        j: deque(
            b
            for b in sorted(vertices, key=lambda _b: (is_doubled[_b], cost_mx[a, _b]))
            if check_transition(a, b, cost_mx, forbidden_val)
        )
        for (a, _), (_, j) in forbidden_transitions
    }
    for v, occurences in occurences_map.items():
        if is_doubled[v] and v not in alternatives.keys():
            for occ in occurences:
                alternatives[v] = deque(
                    b
                    for b in sorted(vertices, key=lambda _b: cost_mx[v, _b])
                    if check_transition(0, b, cost_mx, forbidden_val)
                )

    new_chromosome = chromosome if inplace else chromosome[:]
    additional_iterations = 0
    invalid_trans_ix = 0
    already_inserted = deque()
    # 'reset' further priority queues after fallback to a previous random invalid
    # transition index in the chromosome whose alternatives are not empty yet
    used_alternatives = {j: deque() for j in alternatives.keys()}
    # TODO must iterate over doubled indices also
    while True:
        if additional_iterations > max_additional_iterations:
            return chromosome, rng, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
        invalid_target_ix = forbidden_transitions[invalid_trans_ix][1][1]
        current_alternatives = alternatives[invalid_target_ix]
        current_used_alternatives = used_alternatives[invalid_target_ix]
        try:
            alternative = current_alternatives.popleft()
            while alternative in already_inserted:
                alternative = current_alternatives.popleft()
                current_used_alternatives.appendleft(alternative)
            already_inserted.append(alternative)
        except IndexError:
            if invalid_trans_ix == 0:
                return chromosome, rng, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
            # fallback to some previous invalid place
            already_inserted.pop()
            invalid_trans_ix -= 1
            next_inv_target_ix = forbidden_transitions[invalid_trans_ix][1][1]
            for j in alternatives.keys():
                if j > next_inv_target_ix:
                    alternatives[j].extendleft(used_alternatives[j])
                    used_alternatives[j] = deque()
            continue
        new_chromosome[invalid_target_ix] = alternative
        current_used_alternatives.appendleft(alternative)
        invalid_trans_ix += 1


def fix_tsp_deprecated(
    chromosome: ChromosomeTSP,
    cost_mx: np.ndarray,
    initial_ix: int = 0,
    inplace: bool = False,
) -> tuple[ChromosomeTSP, FixResult]:

    if inplace:
        chromosome = chromosome[:]

    doubled_nodes = find_doubled_indices(chromosome)
    doubles = set(doubled_nodes.keys())
    transitions_possible = check_chromosome(chromosome, cost_mx)

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
            for valid_transition in check_chromosome(chromosome, cost_mx)
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
                for valid_transition in check_chromosome(chromosome, cost_mx)
            )
            return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

        chromosome[repl_ix] = best_replacement
        leftout_vxs.remove(best_replacement)

    return chromosome, FixResult(FixStatus.SUCCESS)


def check_chromosome(
    seq: Sequence[int],
    cost_mx: np.ndarray,
    initial_ix: int = 0,
    forbid_val: Optional[float] = None,
) -> Generator[bool, None, None]:
    return (
        cost >= 0
        and math.isfinite(cost)
        and (True if cost is None else (cost != forbid_val))
        for cost in (
            cost_mx[i, j] for i, j in mit.windowed(it.chain([initial_ix], seq), n=2)
        )
    )
