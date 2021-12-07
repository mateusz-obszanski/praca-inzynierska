from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, Optional, Protocol, TypeVar
from collections.abc import Sequence
import itertools as it
import more_itertools as mit
import math
import sys

import numpy as np

from libs.environment.utils import (
    check_doubles,
    check_transition,
    check_transitions,
    find_invalid_transitions,
)

from libs.optimizers.algorithms.genetic.population.chromosomes import ChromosomeTSP
from libs.utils.iteration import (
    find_all_occurence_indices,
    find_doubled_indices,
)


Rng = TypeVar("Rng", bound=np.random.Generator)


FixStatus = bool


# @dataclass
# class FixResult:
#     fix_status: FixStatus
#     no_of_errors: int = 0

#     def __post_init__(self) -> None:
#         if self.fix_status in (FixStatus.SUCCESS, FixStatus.NO_FIX_NEEDED):
#             if self.no_of_errors:
#                 raise ValueError(
#                     (
#                         "cannot set `fix_status` to `FixStatus.SUCCESS` nor"
#                         "`FixStatus.NO_FIX_NEEDED` and give nonzero `no_of_errors`"
#                     )
#                 )

#             self.no_of_errors = 0


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
    ) -> tuple[Chromosome, FixStatus, Rng]:
        ...


def fix_chromosome_no_doubles(
    chromosome: list[int],
    cost_mx: np.ndarray,
    initial_vx: int,
    forbidden_val: float = -1,
    max_add_iters: int = sys.maxsize,
    inplace: bool = False,
) -> tuple[list[int], FixStatus]:
    """
    :param max_add_iters int: additional iterations for fixing
    """
    if not inplace:
        chromosome = [*chromosome]
    if chromosome[0] != initial_vx:
        chromosome[0] = initial_vx
    if chromosome[-1] != initial_vx:
        chromosome[-1] = initial_vx
    occurences_map = find_all_occurence_indices(chromosome)
    mult_occ_map = {
        vx: occ
        for vx, occ in occurences_map.items()
        if len(occ) > 1 and vx != initial_vx
        # ^^ initial vx should appear at the beginning and the end
    }
    forbidden_transitions = find_invalid_transitions(chromosome, cost_mx, forbidden_val)
    inv_dests = [ft[0][1] for ft in forbidden_transitions]
    inv_dest_ixs = [ft[1][1] for ft in forbidden_transitions]
    mult_vxs = [*mult_occ_map.keys()]
    # mistakes - invalid destinations and multiple-occuring vertices
    mistake_ixs = sorted(
        set(it.chain(inv_dest_ixs, it.chain.from_iterable(mult_occ_map.values())))
    )
    all_vxs = set(range(cost_mx.shape[0]))
    used_vxs = set(chromosome)
    unused_vxs = all_vxs - used_vxs
    alts = tuple(vx for vx in set(it.chain(mult_vxs, inv_dests, unused_vxs)))
    # already used at previous indices to the currently processed mistake index
    already_used_alts: list[Optional[int]] = [None for _ in mistake_ixs]
    already_used_alts_at_mistake_ix = [deque() for _ in mistake_ixs]
    mistake_ix = 0
    iter_n = 0
    mistake_n = len(mistake_ixs)
    max_iter_n = mistake_n + max_add_iters
    while mistake_ix < mistake_n:
        if iter_n >= max_iter_n:
            return chromosome, False
        iter_n += 1
        current_already_used_alts = already_used_alts_at_mistake_ix[mistake_ix]
        unused_alts = [a for a in alts if a not in current_already_used_alts]
        prev_vx_ix = mistake_ixs[mistake_ix] - 1
        prev_vx = chromosome[prev_vx_ix]
        mistake_alts = sorted(unused_alts, key=lambda dest: cost_mx[prev_vx, dest])
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
            # fallback to the previous mistake index
            if mistake_ix <= 0:
                # fallback impossible => fixing impossible
                return chromosome, False
            already_used_alts[mistake_ix] = None
            already_used_alts_at_mistake_ix[mistake_ix] = deque()
            mistake_ix -= 1
            continue
        already_used_alts[mistake_ix] = valid_alt
        current_already_used_alts.append(valid_alt)
        chromosome_mistake_ix = mistake_ixs[mistake_ix]
        chromosome[chromosome_mistake_ix] = valid_alt
        mistake_ix += 1

    return __fix_swap_inv_trans_with_valid(
        chromosome, cost_mx, forbidden_val, initial_vx, max_iter_n, iter_n
    )


def __fix_swap_inv_trans_with_valid(
    chromosome: list[int],
    cost_mx: np.ndarray,
    forbidden_val: float,
    initial_vx: int,
    max_iter_n: int,
    current_iter_n: int,
) -> tuple[list[int], FixStatus]:
    """
    Finishes fixing - only invalid transitions remained but they cannot be
    swapped, hence they must be swapped with some valid genes.
    """

    invalid_transitions = find_invalid_transitions(chromosome, cost_mx, forbidden_val)
    if not invalid_transitions:
        return chromosome, True
    inv_target_chromosome_ixs = [t[1][1] for t in invalid_transitions]
    chromosome_len = len(chromosome)
    the_last_ix = chromosome_len - 1
    valid_vx_ixs = [
        ix
        for ix in range(chromosome_len)
        if ix not in inv_target_chromosome_ixs and 0 != ix != the_last_ix
    ]
    if not valid_vx_ixs:
        # mistakes exist but with no alternatives => fixing impossible
        return chromosome, False
    last_transition_invalid = invalid_transitions[-1][0][1] == initial_vx
    if last_transition_invalid:
        prev_to_last_ix = chromosome_len - 2
        # edge case - destinations are considered invalid, but the last is fixed
        # add second to last index to invalid indices
        if len(inv_target_chromosome_ixs) > 1:
            if inv_target_chromosome_ixs[-2] != prev_to_last_ix:
                inv_target_chromosome_ixs[-1] = prev_to_last_ix
            else:
                inv_target_chromosome_ixs.pop()
        else:
            # length is 1, if it was 0 would return earlier
            inv_target_chromosome_ixs[0] = prev_to_last_ix
        if valid_vx_ixs[-1] == prev_to_last_ix:
            # if present, always at the end
            valid_vx_ixs.pop()

    already_swapped_at_mistake_ix: list[Optional[int]] = [
        None for _ in inv_target_chromosome_ixs
    ]
    already_used_alts_at_mistake_ix = [deque() for _ in inv_target_chromosome_ixs]
    mistake_n = len(inv_target_chromosome_ixs)
    mistake_ix = 0
    while mistake_ix < mistake_n:
        if current_iter_n >= max_iter_n:
            return chromosome, False
        current_iter_n += 1
        current_mistake_chromosome_ix = inv_target_chromosome_ixs[mistake_ix]
        # sorting is needed in case of two mistakes next to each other - the
        # previous affects next
        all_alternative_ixs_at_mistake_ix = __get_sorted_swap_alternative_ixs(
            chromosome,
            valid_vx_ixs,
            cost_mx,
            current_mistake_chromosome_ix,
            forbidden_val=forbidden_val,
        )
        if all_alternative_ixs_at_mistake_ix:
            current_already_used_alt_ixs = already_used_alts_at_mistake_ix[mistake_ix]
            valid_alt_ix = next(
                (
                    aix
                    for aix in all_alternative_ixs_at_mistake_ix
                    if aix not in current_already_used_alt_ixs
                    and chromosome[aix] not in already_swapped_at_mistake_ix
                ),
                None,
            )
        else:
            valid_alt_ix = None
        if valid_alt_ix is None:
            # fallback to the previous mistake index
            if mistake_ix <= 0:
                # fallback impossible => fixing impossible
                return chromosome, False
            already_used_alts_at_mistake_ix[mistake_ix] = deque()
            already_swapped_at_mistake_ix[mistake_ix] = None
            mistake_ix -= 1
            continue
        # swap
        mistake_vx = chromosome[current_mistake_chromosome_ix]
        alt_vx = chromosome[valid_alt_ix]
        chromosome[current_mistake_chromosome_ix] = alt_vx
        chromosome[valid_alt_ix] = mistake_vx
        already_swapped_at_mistake_ix[mistake_ix] = alt_vx
        already_used_alts_at_mistake_ix[mistake_ix].append(valid_alt_ix)
        mistake_ix += 1

    fix_status = check_chromosome_tsp(chromosome, cost_mx, initial_vx, forbidden_val)
    return chromosome, fix_status


def __get_sorted_swap_alternative_ixs(
    chromosome: list[int],
    valid_vx_ixs: list[int],
    cost_mx: np.ndarray,
    i: int,
    forbidden_val: float = -1,
) -> list[int]:
    # valid swap alternative must:
    # - be reachable from before invalid target
    # - have valid connection to one after invalid target
    # in index `i` of valid swap alternative currently invalid vx must:
    # - be reachable from index `i - 1`
    # - have valid connection to one on index `i + 1`
    return sorted(
        (
            vix
            for vix in valid_vx_ixs
            if check_transition(
                chromosome[i - 1], chromosome[vix], cost_mx, forbidden_val
            )
            and check_transition(
                chromosome[vix], chromosome[i + 1], cost_mx, forbidden_val
            )
            and check_transition(
                chromosome[vix - 1], chromosome[i], cost_mx, forbidden_val
            )
            and check_transition(
                chromosome[i], chromosome[vix + 1], cost_mx, forbidden_val
            )
        ),
        key=lambda vix: cost_mx[chromosome[vix - 1], chromosome[i]]
        + cost_mx[chromosome[i], chromosome[vix + 1]]
        + cost_mx[chromosome[i - 1], chromosome[vix]]
        + cost_mx[chromosome[vix], chromosome[i + 1]],
    )


# def fix_chromosome_no_doubles_deprecated(
#     chromosome: list[int],
#     cost_mx: np.ndarray,
#     rng: Rng,
#     forbidden_val: float = -1,
#     max_additional_iterations: int = 10,
#     inplace: bool = False,
# ) -> tuple[list[int], Rng, FixResult]:
#     raise DeprecationWarning
#     occurences_map = find_all_occurence_indices(chromosome)
#     is_doubled = {v: len(occurences) > 1 for v, occurences in occurences_map.items()}
#     # list of (a, b), (ix of a in chromosome, ix of b in chromosome) if forbidden
#     forbidden_transitions = find_invalid_transitions(chromosome, cost_mx, forbidden_val)
#     # create alternatives for substitution of b in invalid transitions a -> b
#     # sorted cost rows filtered by transition validity
#     vertices = list(range(cost_mx.shape[0]))
#     alternatives = {
#         j: deque(
#             b
#             for b in sorted(vertices, key=lambda _b: (is_doubled[_b], cost_mx[a, _b]))
#             if check_transition(a, b, cost_mx, forbidden_val)
#         )
#         for (a, _), (_, j) in forbidden_transitions
#     }
#     for v, occurences in occurences_map.items():
#         if is_doubled[v] and v not in alternatives.keys():
#             for occ in occurences:
#                 alternatives[v] = deque(
#                     b
#                     for b in sorted(vertices, key=lambda _b: cost_mx[v, _b])
#                     if check_transition(0, b, cost_mx, forbidden_val)
#                 )

#     new_chromosome = chromosome if inplace else [*chromosome]
#     additional_iterations = 0
#     invalid_trans_ix = 0
#     already_inserted = deque()
#     # 'reset' further priority queues after fallback to a previous random invalid
#     # transition index in the chromosome whose alternatives are not empty yet
#     used_alternatives = {j: deque() for j in alternatives.keys()}
#     # TODO must iterate over doubled indices also
#     while True:
#         if additional_iterations > max_additional_iterations:
#             return chromosome, rng, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
#         invalid_target_ix = forbidden_transitions[invalid_trans_ix][1][1]
#         current_alternatives = alternatives[invalid_target_ix]
#         current_used_alternatives = used_alternatives[invalid_target_ix]
#         try:
#             alternative = current_alternatives.popleft()
#             while alternative in already_inserted:
#                 alternative = current_alternatives.popleft()
#                 current_used_alternatives.appendleft(alternative)
#             already_inserted.append(alternative)
#         except IndexError:
#             if invalid_trans_ix == 0:
#                 return chromosome, rng, FixResult(FixStatus.FAILURE, no_of_errors=(-1))
#             # fallback to some previous invalid place
#             already_inserted.pop()
#             invalid_trans_ix -= 1
#             next_inv_target_ix = forbidden_transitions[invalid_trans_ix][1][1]
#             for j in alternatives.keys():
#                 if j > next_inv_target_ix:
#                     alternatives[j].extendleft(used_alternatives[j])
#                     used_alternatives[j] = deque()
#             continue
#         new_chromosome[invalid_target_ix] = alternative
#         current_used_alternatives.appendleft(alternative)
#         invalid_trans_ix += 1


# def fix_tsp_deprecated(
#     chromosome: ChromosomeTSP,
#     cost_mx: np.ndarray,
#     initial_ix: int = 0,
#     inplace: bool = False,
# ) -> tuple[ChromosomeTSP, FixResult]:

#     if not inplace:
#         chromosome = [*chromosome]

#     doubled_nodes = find_doubled_indices(chromosome)
#     doubles = set(doubled_nodes.keys())
#     transitions_possible = check_transitions_deprecated(chromosome, cost_mx)

#     if not doubles and all(transitions_possible):
#         return chromosome, FixResult(FixStatus.SUCCESS)

#     all_vxs = set(range(cost_mx.shape[0]))
#     leftout_vxs = all_vxs - doubles
#     all_double_ixs = it.chain.from_iterable(doubled_nodes.values())
#     visited_nodes = list(it.chain([initial_ix], chromosome))

#     # replace with the best replacements
#     for doubled_ix in all_double_ixs:
#         prev_vx = visited_nodes[doubled_ix - 1]
#         try:
#             # If two vertices have the same cost, the smaller one
#             # (they are ints) is chosen.
#             _, best_replacement = min(
#                 (cost, vx)
#                 for vx in leftout_vxs
#                 if (cost := cost_mx[prev_vx, vx]) > 0 and math.isfinite(cost)
#             )

#         except ValueError:
#             # replacement is impossible (in place of current double)
#             # it will be attempted later in place of the original genetry:
#             continue

#         previously_double = chromosome[doubled_ix]
#         chromosome[doubled_ix] = best_replacement
#         leftout_vxs.remove(best_replacement)
#         doubles.remove(previously_double)

#     if not leftout_vxs:
#         no_of_errors = sum(
#             not valid_transition
#             for valid_transition in check_transitions_deprecated(chromosome, cost_mx)
#         )

#         if no_of_errors == 0:
#             return chromosome, FixResult(FixStatus.SUCCESS)

#         return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

#     # there were impossible replacements - attempt to replace in place
#     # of the original genes

#     all_vxs = list(all_vxs)

#     for left_double in doubles:
#         repl_ix = next(ix for ix, vx in enumerate(all_vxs) if vx == left_double)
#         if repl_ix == 0:
#             # the first element is the starting point - can not change that
#             return chromosome, FixResult(FixStatus.FAILURE)

#         prev_vx = repl_ix - 1

#         try:
#             _, best_replacement = min(
#                 (cost, vx)
#                 for vx in leftout_vxs
#                 if (cost := cost_mx[prev_vx, vx]) > 0 and math.isfinite(cost)
#             )
#         except ValueError:
#             # min got empty iterator - no possible replacements
#             # doubles mapping has not been mutated so it is used
#             no_of_errors = len(doubled_nodes) + sum(
#                 not valid_transition
#                 for valid_transition in check_transitions_deprecated(
#                     chromosome, cost_mx
#                 )
#             )
#             return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

#         chromosome[repl_ix] = best_replacement
#         leftout_vxs.remove(best_replacement)

#     return chromosome, FixResult(FixStatus.SUCCESS)


def check_transitions_deprecated(
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


def check_chromosome_tsp(
    chromosome: list[int],
    cost_mx: np.ndarray,
    initial_vx: int = 0,
    forbidden_val: float = -1,
) -> bool:
    return check_transitions(
        chromosome, cost_mx, forbid_val=forbidden_val
    ) and check_doubles(chromosome, initial_vx)
